import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of encoder.")
    parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--image_shape', type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path_to_imagefolder', type=str, default="", help='path to imagefolder of labeled data')
    parser.add_argument('--destination', type=str, default="", help='path to save the results')
    parser.add_argument('--num_bins', type=int, default=25, help='number of bins for calibration plot')
    args = parser.parse_args()
        
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
    
    print("Extracting features...")

    X_tr = model.predict(ds_train)
    y_tr = np.concatenate([y for x, y in ds_train], axis=0)
    X_te = model.predict(ds_val)
    y_te = np.concatenate([y for x, y in ds_val], axis=0)

    print("Fitting logistic regression model...")
    
    log_model = LogisticRegression(
        random_state=parser.parse_args().seed,
        max_iter=10000,
        multi_class="multinomial",
        class_weight="balanced",
    )
    
    log_model.fit(X_tr, y_tr)
    
    probs = log_model.predict_proba(X_te)
    labs = np.zeros_like(probs)
    labs[np.arange(len(y_te)), y_te] = 1

    bins = np.linspace(0, 1, args.num_bins)
    step = bins[1] - bins[0]
    steps = bins - step / 2
    steps[0] = 0
    low = bins[:-1]
    upp = bins[1:]
            
    p = np.zeros(len(low))
    freqs = np.zeros(len(low))
    observed = np.zeros(len(low))
        
    for i in range(len(low)):
        _labs = labs[np.where((probs >= low[i]) * (probs < upp[i]))]
        p[i] = probs[np.where((probs >= low[i]) * (probs < upp[i]))].mean()
        freqs[i] = _labs.mean()
        observed[i] = _labs.shape[0]
    
    error = np.abs(freqs - p).mean()
    
    
    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, np.log(observed), width=step*0.95, color="b")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observations (log)")
    plt.savefig(os.path.join(args.destination, "observations.png"), dpi=300)
    
    
    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, freqs, width=step*0.95, color="b")
    plt.step(bins, np.concatenate([[0], p]), where="pre", color="k", linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration error: {error:.3f}")
    plt.savefig(os.path.join(args.destination, "calibration.png"), dpi=300)
    


