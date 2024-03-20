import os
import sys
import glob

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of encoder.")
    parser.add_argument('--image_shape', type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path_to_imagefolder', type=str, default="/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20_classes", help='path to imagefolder of labeled data')
    parser.add_argument('--destination', type=str, default="", help='path to save the results')
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
    
    print("Loading encoders...")
    
    weights = glob.glob("trained models/Xception/*.h5")
    weights += ["imagenet"]
    encoders = [tf.keras.applications.Xception(
        weights=w,
        include_top=False, 
        pooling="avg", 
        input_shape=parser.parse_args().image_shape, 
        ) for w in weights]
    models = [tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0), encoder]) for encoder in encoders]

    os.makedirs(args.destination, exist_ok=True)
    
    print("Extracting features...")

    class_names = ds_train.class_names
    
    preds = np.zeros((len(models), 140, len(class_names)))
    
    neighbors = [1, 3, 5, 7, 9]
    k_preds = np.zeros((len(models), 140, len(neighbors)))
    
    accuracies = pd.DataFrame()
    
    
    for i, model in enumerate(models):
    
        X_tr = model.predict(ds_train)
        y_tr = np.concatenate([y for x, y in ds_train], axis=0)
        X_te = model.predict(ds_val)
        y_te = np.concatenate([y for x, y in ds_val], axis=0)
    
    
        print("Linear evaluation...")


        log_model = LogisticRegression(
            random_state=parser.parse_args().seed,
            max_iter=10000,
            multi_class="multinomial",
            class_weight="balanced",
        )
    
        log_model.fit(X_tr, y_tr)
        y_pred = log_model.predict_proba(X_te)
        preds[i] = y_pred
        accuracies.loc[i, "model"] = weights[i]
        accuracies.loc[i, "accuracy"] = accuracy_score(y_te, np.argmax(y_pred, axis=1))
        
        for j, k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr, y_tr)
            k_pred = knn.predict(X_te)
            k_preds[i, :, j] = k_pred
            accuracies.loc[i, f"{k}-nn_accuracy"] = accuracy_score(y_te, k_pred)
        
    accuracies.to_csv(os.path.join(args.destination, "accuracies.csv"))
    
    # sort the models by accuracy
    idxs = np.argsort(accuracies["accuracy"])[::-1]
    
    
    
    for i in range(len(models)):
        idx = idxs[:i+1]
        
        y_pred_proba = preds[idx].mean(axis=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # compute summary metrics on the test set
        summary_table = pd.DataFrame()
    
        summary_table.loc["logistic", "log_loss"] = log_loss(y_te, y_pred_proba)
        summary_table.loc["logistic", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
        summary_table.loc["logistic", "accuracy"] = accuracy_score(y_te, y_pred)
        summary_table.loc["logistic", "mean_precision"] = precision_score(y_te, y_pred, average="macro")

        for j, k in enumerate(neighbors):
        
            # find majority vote
            k_pred = mode(k_preds[idx, :, j], axis=0)[0].squeeze()
            summary_table.loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(y_te, k_pred)
            summary_table.loc[f"k={k}", "accuracy"] = accuracy_score(y_te, k_pred)
            summary_table.loc[f"k={k}", "mean_precision"] = precision_score(y_te, k_pred, average="macro")
        
        summary_table.to_csv(os.path.join(args.destination, f"summary_table_ensemble_{i + 1}.csv"))