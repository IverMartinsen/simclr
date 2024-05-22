import os
import sys

sys.path.append(os.getcwd())

import argparse
import glob
import json
import tensorflow as tf
from simclr import ContrastiveModel, get_projection_head
from transforms import RandomColorAffine, RandomBlur
from random_crop_v2 import RandomCrop, random_crop_with_resize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--input_shape", type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--loss_implementation", type=str, default="simple")
    parser.add_argument("--path_to_files", type=str, default="./data/NO 6407-6-5/100K_BENCHMARK_224x224/images/")
    parser.add_argument("--job_id", type=str, default="simclr")
    args = parser.parse_args()
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU') )}")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"Name: {gpu}, Type: {gpu.device_type}")
    
    files = glob.glob(args.path_to_files + '*tfrecord*')
    dataset_info = json.load(open(os.path.join(args.path_to_files, 'dataset_info.json')))
    shard_lengths = [int(x) for x in dataset_info['splits'][0]['shardLengths']]
    dataset_size = sum(shard_lengths)
    
    assert len(files) == len(shard_lengths)

    print(f"Currently in directory: {os.getcwd()}")
    print(f"Found {len(files)} tfrecord files")
    print(f"Dataset size: {dataset_size}")

    def _tfrecord_map_function(x):
        """Parse a single image from a tfrecord file."""
        # Dict with key 'image' and value of type string
        x = tf.io.parse_single_example(x, {"image": tf.io.FixedLenFeature([], tf.string)})
        x = tf.io.decode_jpeg(x["image"], channels=3)
        x = tf.ensure_shape(x, [224, 224, 3]) # a hack to set the shape attribute
        x = tf.image.resize(x, args.input_shape[:2])
        x = x / 255.0
        return x
    
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(args.buffer_size)
    dataset = dataset.map(_tfrecord_map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    destination_folder = "./trained_models/" + args.job_id + "/"
    
    os.makedirs(destination_folder, exist_ok=True)
    
    # save the arguments
    with open(destination_folder + "args.txt", "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        
    class SaveModelCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_freq, folder):
            super(SaveModelCallback, self).__init__()

            self.log_freq = log_freq
            self.folder = folder
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.log_freq == 0:
                self.model.encoder.save_weights(self.folder + "checkpoint" + f"{epoch:04d}.h5")            
    
    class SaveLearningRate(tf.keras.callbacks.Callback):
        def __init__(self, folder):
            super(SaveLearningRate, self).__init__()
            
            self.folder = folder
            self.learning_rates = []
            
        def on_epoch_end(self, epoch, logs=None):
            self.learning_rates.append(self.model.optimizer.learning_rate.numpy())
            
            with open(self.folder + "learning_rates.json", "w") as f:
                json.dump(self.learning_rates, f)
    
    callbacks = []
    callbacks.append(SaveModelCallback(log_freq=10, folder=destination_folder))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=destination_folder))

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-3 * args.batch_size / 128, # 1e-3 is the default learning rate
            decay_steps=dataset_size // args.batch_size, # decay on every epoch
            decay_rate=0.05,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(schedule)
        
        min_area = 0.25
        
        augmenter = tf.keras.Sequential([
            tf.keras.Input(shape=args.input_shape),
            tf.keras.layers.RandomFlip("horizontal"),
            RandomCrop(height=args.input_shape[0], width=args.input_shape[1]),
            RandomColorAffine(brightness=0.6, jitter=0.2),
            RandomBlur(p=0.5, kernel_size=9),
        ])
        
        model = ContrastiveModel(
            augmenter = augmenter,
            encoder = tf.keras.applications.Xception(include_top=False, pooling="avg", input_shape=args.input_shape, weights=None), 
            projection_head = get_projection_head(width=128, input_shape=(2048, )),
            temperature = args.temperature,
            loss_implementation = args.loss_implementation,
        )
        
        model.compile(optimizer=optimizer, probe_optimizer=None)
    
    print("Augmenter summary:")
    model.augmenter.summary()
    
    history = model.fit(
        dataset, 
        epochs=args.epochs,
        steps_per_epoch=None,
        validation_data=None,
        callbacks=callbacks,
        )
    
    print("Training finished. Saving the model and history.")
    
    with open(destination_folder + "history.json", "w") as f:
        json.dump(history.history, f)
    
    model.encoder.save_weights(destination_folder + "encoder.h5")
