import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
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
    parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--image_shape', type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path_to_imagefolder', type=str, default="", help='path to imagefolder of labeled data')
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

    model_id = args.pretrained_weights.split(".")[0]
    
    destination = os.path.join("trained models", model_id)
    os.makedirs(destination, exist_ok=True)
    
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
    
    class_weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=np.concatenate([y_tr, y_te]))
    
    # compute summary metrics on the test set
    y_pred = log_model.predict(X_te)
    summary_table.loc["logistic", "log_loss"] = log_loss(y_te, log_model.predict_proba(X_te))
    summary_table.loc["logistic", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
    summary_table.loc["logistic", "accuracy"] = accuracy_score(y_te, y_pred)
    summary_table.loc["logistic", "mean_precision"] = precision_score(y_te, y_pred, average="macro")

    # add the metrics to the table
    table = pd.DataFrame()
    table["class"] = class_names
    table["precision"] = precision_score(y_te, y_pred, average=None)
    table["recall"] = recall_score(y_te, y_pred, average=None)
    table["f1"] = f1_score(y_te, y_pred, average=None)
    table.to_csv(os.path.join(destination, "table_logistic.csv"))

    # fit a knn model for each k
    for k in range(1, 10, 2):
        
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)
        
        # compute the accuracy and balanced accuracy and mean precision on the test set
        summary_table.loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
        summary_table.loc[f"k={k}", "accuracy"] = accuracy_score(y_te, y_pred)
        summary_table.loc[f"k={k}", "mean_precision"] = precision_score(y_te, y_pred, average="macro")

        # add the metrics to the table
        table = pd.DataFrame()
        table["class"] = class_names
        table["precision"] = precision_score(y_te, y_pred, average=None)
        table["recall"] = recall_score(y_te, y_pred, average=None)
        table["f1"] = f1_score(y_te, y_pred, average=None)
        table.to_csv(os.path.join(destination, f"table_k{k}.csv"))
    
    summary_table.to_csv(os.path.join(destination, "summary_table.csv"))
