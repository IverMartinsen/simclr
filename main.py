import os
import sys
from typing import Any

sys.path.append(os.getcwd())

import argparse
import glob
import wandb
import tensorflow as tf
from datetime import datetime
from simclr import ContrastiveModel, get_augmenter, get_encoder, get_projection_head
#from scampi_unsupervised.tf_utils import LogisticRegressionCallback
#from scampi_evaluation.prepare_labelled_data import get_numpy_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--dataset_size", type=int, default=200000)
    parser.add_argument("--input_shape", type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--loss_implementation", type=str, default="simple")
    parser.add_argument("--path_to_files", type=str, default="./data/NO 6407-6-5/100K_BENCHMARK_224x224/images/")
    args = parser.parse_args()
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    #dataset = tf.data.Dataset.list_files(args.path_to_files + "*.png")
    #dataset = dataset.shuffle(args.buffer_size)
    #dataset = dataset.map(lambda x: tf.io.read_file(x))
    #dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3))
    #dataset = dataset.map(lambda x: tf.image.resize(x, args.input_shape[:2]))
    #dataset = dataset.batch(args.batch_size, drop_remainder=True)
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    def _tfrecord_map_function(x):
        """Parse a single image from a tfrecord file."""
        # Dict with key 'image' and value of type string
        x = tf.io.parse_single_example(x, {"image": tf.io.FixedLenFeature([], tf.string)})
        # Tensor of type uint8
        x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)
        x = tf.ensure_shape(x, [224, 224, 3])
        return x
    
    dataset = tf.data.TFRecordDataset(glob.glob(args.path_to_files + "*.tfrecords"))
    dataset = dataset.shuffle(args.buffer_size)
    dataset = dataset.map(_tfrecord_map_function)
    dataset = dataset.map(lambda x: tf.image.resize(x, args.input_shape[:2]))
    dataset = dataset.map(lambda x: x / 255.0)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    
    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_id = "simclr_" + timestr
    
    destination_folder = "./scampi_unsupervised/frameworks/simclr/" + train_id + "/"
    
    os.makedirs(destination_folder, exist_ok=True)
    
    wandb.init(project="scampi", name=train_id, config=vars(args))
    
    model = ContrastiveModel(
        augmenter = get_augmenter(input_shape=args.input_shape, min_area=0.25, brightness=0.6, jitter=0.2),
        encoder = tf.keras.applications.Xception(include_top=False, pooling="avg", input_shape=args.input_shape, weights=None), 
        projection_head = get_projection_head(width=128, input_shape=(2048, )),
        temperature = args.temperature,
        loss_implementation = args.loss_implementation,
    )
    
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=1e-3,
        decay_steps=args.dataset_size // args.batch_size,
        decay_rate=0.05,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(schedule)
    
    model.compile(optimizer=optimizer, probe_optimizer=None)


    class SaveModelCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_freq, folder):
            super(SaveModelCallback, self).__init__()

            self.log_freq = log_freq
            self.folder = folder
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.log_freq == 0:
                self.model.encoder.save_weights(self.folder + "checkpoint" + f"{epoch:02d}.h5")

    
    callbacks = []
    callbacks.append(wandb.keras.WandbCallback(save_model=False))
    callbacks.append(SaveModelCallback(log_freq=10, folder=destination_folder))
    
    print("Augmenter summary:")
    model.augmenter.summary()
    
    history = model.fit(
        dataset, 
        epochs=args.epochs,
        steps_per_epoch=None,
        validation_data=None,
        callbacks=callbacks,
        )
    
    model.encoder.save_weights(destination_folder + "encoder.h5")
