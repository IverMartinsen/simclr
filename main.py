import os
import sys

sys.path.append(os.getcwd())

import argparse
import wandb
import tensorflow as tf
from datetime import datetime
from scampi_unsupervised.frameworks.simclr.simclr import ContrastiveModel, get_augmenter, get_encoder, get_projection_head, Augmenter
#from scampi_unsupervised.dataloader import get_tfrecord_dataset, get_tfrecord_benchmark_dataset, get_dataset_from_hdf5_old
from scampi_unsupervised.tf_utils import LogisticRegressionCallback
from scampi_evaluation.prepare_labelled_data import get_numpy_dataset

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
    
    dataset = tf.data.Dataset.list_files(parser.parse_args().path_to_files + "*.png")
    dataset = dataset.shuffle(parser.parse_args().buffer_size)
    dataset = dataset.map(lambda x: tf.io.read_file(x))
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3))
    dataset = dataset.map(lambda x: tf.image.resize(x, parser.parse_args().input_shape[:2]))
    dataset = dataset.batch(parser.parse_args().batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
    #dataset = get_tfrecord_dataset(batch_size=parser.parse_args().batch_size, buffer_size=parser.parse_args().buffer_size)
    #dataset = get_tfrecord_benchmark_dataset(batch_size=parser.parse_args().batch_size, buffer_size=parser.parse_args().buffer_size, dataset_size=parser.parse_args().dataset_size)
    #dataset = get_dataset_from_hdf5_old(image_shape=parser.parse_args().input_shape, batch_size=parser.parse_args().batch_size, dataset_size=parser.parse_args().dataset_size)
    dataset_labelled = get_numpy_dataset(num_classes=10, splits=[0.8, 0.2], seed=1234, img_shape=parser.parse_args().input_shape)
    
    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_id = "simclr_" + timestr
    
    wandb.init(project="scampi", name=train_id, config=vars(parser.parse_args()))
    
    model = ContrastiveModel(
        augmenter = get_augmenter(input_shape=parser.parse_args().input_shape, min_area=0.25, brightness=0.6, jitter=0.2),
        encoder = tf.keras.applications.Xception(include_top=False, pooling="avg", input_shape=parser.parse_args().input_shape, weights=None), 
        projection_head = get_projection_head(width=128, input_shape=(2048, )),
        temperature = parser.parse_args().temperature,
        loss_implementation = parser.parse_args().loss_implementation,
    )
    
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=1e-3,
        decay_steps=parser.parse_args().dataset_size // parser.parse_args().batch_size,
        decay_rate=0.05,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(schedule)
    
    model.compile(optimizer=optimizer, probe_optimizer=None)

    callbacks = []
    callbacks.append(wandb.keras.WandbCallback(save_model=False))
    #callbacks.append(LogisticRegressionCallback(dataset_labelled, log_freq=5))
    
    print("Augmenter summary:")
    model.augmenter.summary()
    
    history = model.fit(
        dataset, 
        epochs=parser.parse_args().epochs,
        steps_per_epoch=None,
        validation_data=None,
        callbacks=callbacks,
        )
    
    model.encoder.save_weights("./scampi_unsupervised/frameworks/simclr/" + train_id + "_encoder.h5")
