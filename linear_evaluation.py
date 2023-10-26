import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
from datetime import datetime
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
from scampi_evaluation.prepare_labelled_data import get_numpy_dataset, get_dataset_stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of encoder.")
    #parser.add_argument('--method', type=str, default='logistic', help='logistic or knn')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--image_shape', type=int, nargs="+", default=[224, 224, 3])
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--validation_split', type=float, default=0.2)
    args = parser.parse_args()
    
    print("Loading data...")

    (x_tr, y_tr, _, _), (x_te, y_te, _, _) = get_numpy_dataset(
        num_classes=args.num_classes,
        img_shape=args.image_shape,
        splits=(1 - args.validation_split, args.validation_split),
        seed=args.seed,
    )

    class_names, _, _ = get_dataset_stats(num_classes=args.num_classes)

    # discard the last class (background)
    x_tr = x_tr[y_tr != args.num_classes - 1]
    y_tr = y_tr[y_tr != args.num_classes - 1]
    x_te = x_te[y_te != args.num_classes - 1]
    y_te = y_te[y_te != args.num_classes - 1]
    class_names = class_names[:-1]
    
    print("Loading encoder...")
    
    encoder = tf.keras.applications.Xception(
        weights=args.pretrained_weights,
        include_top=False, 
        pooling="avg", 
        input_shape=parser.parse_args().image_shape, 
        )
    
    model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0), encoder])

    model_id = args.pretrained_weights.split(".")[0]
    
    path_to_simclr = "./scampi_unsupervised/frameworks/simclr/"
    destination = os.path.join(path_to_simclr, model_id)
    os.makedirs(destination, exist_ok=True)
    
    summary_table = pd.DataFrame()
    #summary_table.index = [model_id]
    #summary_table["model_id"] = model_id

    print("Extracting features...")

    X_tr = model.predict(x_tr)
    X_te = model.predict(x_te)

    print("Linear evaluation...")

    #if parser.parse_args().method == "logistic":
        # train a logistic regression model and compute the log loss
    log_model = LogisticRegression(
        random_state=parser.parse_args().seed,
        max_iter=10000,
        multi_class="multinomial",
        class_weight="balanced",
    )
    
    log_model.fit(X_tr, y_tr)
    
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_tr), y=np.concatenate([y_tr, y_te])
    )
    sample_weights = np.array([class_weights[y] for y in y_te])
    
    # compute the log loss on the test set
    y_pred = log_model.predict(X_te)
    
    #summary_table.loc[model_id, "weighted_log_loss"] = log_loss(
    #    y_te, log_model.predict_proba(X_te), sample_weight=sample_weights
    #)
    
    summary_table.loc["logistic", "log_loss"] = log_loss(y_te, log_model.predict_proba(X_te))

    #summary_table.loc[model_id, "relevant_log_loss"] = log_loss(y_te, log_model.predict_proba(X_te), sample_weight=y_te != parser.parse_args().num_classes - 1)
    
    # ...the same for the training set
    #y_pred_train = log_model.predict(X_tr)

    #summary_table.loc[model_id, "weighted_log_loss_train"] = log_loss(
    #    y_tr,
    #    log_model.predict_proba(X_tr),
    #    sample_weight=np.array([class_weights[y] for y in y_tr]),
    #)

    #summary_table.loc[model_id, "log_loss_train"] = log_loss(
    #    y_tr, log_model.predict_proba(X_tr)
    #)

    # compute the accuracy and balanced accuracy and mean precision on the test set
    summary_table.loc["logistic", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
    summary_table.loc["logistic", "accuracy"] = accuracy_score(y_te, y_pred)
    summary_table.loc["logistic", "relevant_accuracy"] = accuracy_score(y_te, y_pred, sample_weight = y_te != parser.parse_args().num_classes - 1)
    summary_table.loc["logistic", "mean_precision"] = precision_score(y_te, y_pred, average="macro")

    # ...the same for the training set
    #summary_table.loc[model_id, "balanced_accuracy_train"] = balanced_accuracy_score(y_tr, y_pred_train)
    #summary_table.loc[model_id, "accuracy_train"] = accuracy_score(y_tr, y_pred_train)
    #summary_table.loc[model_id, "mean_precision_train"] = precision_score(y_tr, y_pred_train, average="macro")

    # add the metrics to the table
    table = pd.DataFrame()
    table["class"] = class_names

    table["precision"] = precision_score(y_te, y_pred, average=None)
    table["recall"] = recall_score(y_te, y_pred, average=None)
    table.to_csv(os.path.join(destination, "table_logistic.csv"))

    #elif parser.parse_args().method == "knn":
        # train a k-nearest neighbors model
    for k in range(1, 10, 2):


        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)
        y_pred_train = knn.predict(X_tr)

        # compute the accuracy and balanced accuracy and mean precision on the test set
        summary_table.loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
        summary_table.loc[f"k={k}", "accuracy"] = accuracy_score(y_te, y_pred)
        summary_table.loc[f"k={k}", "relevant_accuracy"] = accuracy_score(y_te, y_pred, sample_weight = y_te != parser.parse_args().num_classes - 1)
        summary_table.loc[f"k={k}", "mean_precision"] = precision_score(y_te, y_pred, average="macro")

        # # ...the same for the training set
        # summary_table.loc[model_id, "balanced_accuracy_train"] = balanced_accuracy_score(y_tr, y_pred_train)
        # summary_table.loc[model_id, "accuracy_train"] = accuracy_score(y_tr, y_pred_train)
        # summary_table.loc[model_id, "mean_precision_train"] = precision_score(y_tr, y_pred_train, average="macro")

        # add the metrics to the table
        table = pd.DataFrame()
        table["class"] = class_names

        table["precision"] = precision_score(y_te, y_pred, average=None)
        table["recall"] = recall_score(y_te, y_pred, average=None)
        
        table.to_csv(os.path.join(destination, f"table_k{k}.csv"))

    # log confusion matrix
    #conf_mat = pd.DataFrame(columns=class_names, index=class_names)
    #conf_mat.iloc[:, :] = tf.math.confusion_matrix(
    #    y_te, y_pred, num_classes=parser.parse_args().num_classes
    #).numpy()

    #wandb.log({model_id + "_confusion_matrix": wandb.Table(dataframe=conf_mat)})
    #wandb.log({"table": wandb.Table(dataframe=table)})
    #wandb.log({"summary_table": wandb.Table(dataframe=summary_table)})
    #
    #wandb.finish()
    
    # save csv
    
    summary_table.to_csv(os.path.join(destination, "summary_table.csv"))
