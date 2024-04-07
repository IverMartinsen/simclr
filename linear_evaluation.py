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
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of encoder.")
    parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--image_shape', type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path_to_imagefolder', type=str, default="", help='path to imagefolder of labeled data')
    parser.add_argument('--destination', type=str, default="", help='path to save the results')
    args = parser.parse_args()
    
    print("Loading data...")
    
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.path_to_imagefolder,
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
    
    os.makedirs(args.destination, exist_ok=True)
    
    print("Extracting features...")
    
    X = model.predict(ds)
    y = np.concatenate([y for x, y in ds], axis=0)
    
    class_names = ds.class_names
    
    summary_tables = [pd.DataFrame() for _ in range(10)]

    for seed in range(10):
        print(f"Evaluating seed {seed}...")
        # create a train and test split
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        log_model = LogisticRegression(
            random_state=seed,
            max_iter=10000,
            multi_class="multinomial",
            class_weight="balanced",
        )
    
        log_model.fit(X_tr, y_tr)
        
        y_pred = log_model.predict(X_te)
        
        summary_tables[seed].loc["logistic", "log_loss"] = log_loss(y_te, log_model.predict_proba(X_te))
        summary_tables[seed].loc["logistic", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
        summary_tables[seed].loc["logistic", "accuracy"] = accuracy_score(y_te, y_pred)
        summary_tables[seed].loc["logistic", "mean_precision"] = precision_score(y_te, y_pred, average="macro")
        
        # fit a knn model for each k
        for k in range(1, 10, 2):
            
            knn = KNeighborsClassifier(n_neighbors=k, p=2)
            knn.fit(X_tr, y_tr)
            y_pred = knn.predict(X_te)
            
            # compute the accuracy and balanced accuracy and mean precision on the test set
            summary_tables[seed].loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
            summary_tables[seed].loc[f"k={k}", "accuracy"] = accuracy_score(y_te, y_pred)
            summary_tables[seed].loc[f"k={k}", "mean_precision"] = precision_score(y_te, y_pred, average="macro")
                
    mean_summary_table = pd.concat(summary_tables).groupby(level=0).mean()
    mean_summary_table.to_csv(os.path.join(args.destination, "mean_summary_table.csv"))
    for i, summary_table in enumerate(summary_tables):
        summary_table.to_csv(os.path.join(args.destination, f"summary_table_seed{i}.csv"))
