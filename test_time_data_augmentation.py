import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression

# run tensorflow in eager mode
tf.config.run_functions_eagerly(True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of encoder.")
    parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--image_shape', type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path_to_imagefolder', type=str, default="", help='path to imagefolder of labeled data')
    parser.add_argument('--destination', type=str, default="", help='path to save the results')
    args = parser.parse_args()
    
    args.pretrained_weights = "./trained models/simclr_20240201_141701/_encoder.h5"
    args.pretrained_weights = "./trained models/simclr_20240117_143104/_encoder.h5"
    args.path_to_imagefolder = "/Users/ima029/SCAMPI DATA/labelled crops (NPD)/genera_level/imagefolder50classes"
    
    print("Loading data...")

    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.path_to_imagefolder, "train"),
        image_size=args.image_shape[:2],
        shuffle=False,
    )
    
    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.path_to_imagefolder, "val"),
        image_size=args.image_shape[:2],
        shuffle=False,
    )
        
    print("Loading encoder...")
    
    encoder = tf.keras.applications.Xception(
        weights=args.pretrained_weights,
        include_top=False, 
        pooling="avg", 
        input_shape=parser.parse_args().image_shape, 
        )
    
    model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0), encoder])

    #model_id = args.pretrained_weights.split(".")[0]
    
    #destination = os.path.join("trained models", model_id)
    os.makedirs(args.destination, exist_ok=True)
    
    print("Extracting features...")

    X_tr = model.predict(ds_train)
    y_tr = np.concatenate([y for x, y in ds_train], axis=0)
    X_te = model.predict(ds_val)
    y_te = np.concatenate([y for x, y in ds_val], axis=0)
    
    class_names = ds_train.class_names
    
    print("Linear evaluation...")

    summary_table = pd.DataFrame()
    
    log_model = LogisticRegression(
        random_state=parser.parse_args().seed,
        max_iter=10000,
        multi_class="multinomial",
        class_weight="balanced",
    )
    
    log_model.fit(X_tr, y_tr)

from transforms import RandomBlur
from transforms import random_blur, color_drop


tf.config.run_functions_eagerly(False)

vectors = []

seed = np.random.randint(0, 1000), np.random.randint(0, 1000)

for batch in ds_val:
    x = batch[0] / 255.0
    #x = random_blur(x)
    #x = tf.image.stateless_random_flip_left_right(x, seed=seed)
    #x = tf.image.stateless_random_flip_up_down(x, seed=seed)
    #x = tf.image.stateless_random_brightness(x, 0.2, seed=seed)
    #x = tf.image.stateless_random_contrast(x, 0.2, 0.8, seed=seed)
    #x = tf.image.stateless_random_saturation(x, 0.2, 0.8, seed=seed)
    #x = tf.image.stateless_random_jpeg_quality(x[0], 75, 95, seed=seed)
    #x = color_drop(x)
    
    x = encoder(x)
    vectors.append(x)

X_te = np.concatenate(vectors, axis=0)

log_model.score(X_te, y_te)