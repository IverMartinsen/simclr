import os
import sys

sys.path.append(os.getcwd())

import argparse
import glob
import json
import tensorflow as tf


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
        return x
    
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(args.buffer_size)
    dataset = dataset.map(_tfrecord_map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.image.resize(x, args.input_shape[:2]))
    dataset = dataset.map(lambda x: x / 255.0)
    dataset = dataset.map(lambda x: (x, 1)) # add a label
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    destination_folder = "./trained_models/" + args.job_id + "/"
    
    os.makedirs(destination_folder, exist_ok=True)
    
    # save the arguments
    with open(destination_folder + "args.txt", "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    #wandb.init(project="scampi", name=train_id, config=vars(args))    
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    callbacks = []
    #callbacks.append(wandb.keras.WandbCallback(save_model=False))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=destination_folder))

    # Open a strategy scope.
    with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
        from simclr import ContrastiveModel, get_augmenter, get_projection_head
        from random_crop import RandomCrop
        from transforms import RandomColorAffine, RandomBlur
        
        schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-3,
            decay_steps=dataset_size // args.batch_size,
            decay_rate=0.05,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(schedule)
        
        min_area = 0.25
        zoom_factor = 1.0 - tf.sqrt(min_area)
        
        model = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            RandomCrop(height=args.input_shape[0], width=args.input_shape[1], min_area=min_area),
            RandomColorAffine(brightness=0.6, jitter=0.2),
            RandomBlur(p=0.5, kernel_size=9),
            tf.keras.applications.Xception(include_top=False, pooling="avg", input_shape=args.input_shape, weights=None),
            tf.keras.layers.Dense(1),
        ])
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        
    history = model.fit(
        dataset, 
        epochs=args.epochs,
        steps_per_epoch=None,
        validation_data=None,
        callbacks=callbacks,
        )
    
    # save the history
    with open(destination_folder + "history.json", "w") as f:
        json.dump(history.history, f)
    
    model.encoder.save_weights(destination_folder + "encoder.h5")
