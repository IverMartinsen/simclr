import os
import sys

sys.path.append(os.getcwd())

import wandb
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
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from scampi_unsupervised.models import add_rescaling_layer
from scampi_evaluation.prepare_labelled_data import get_numpy_dataset, get_dataset_stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of encoders.")
    parser.add_argument('--method', type=str, default='logistic', help='logistic or knn')
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--image_shape', type=int, nargs="+", default=[96, 96, 3])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--validation_split', type=float, default=0.2)

    print("Loading data...")

    (x_tr, y_tr, _, _), (x_te, y_te, _, _) = get_numpy_dataset(
        num_classes=parser.parse_args().num_classes,
        img_shape=parser.parse_args().image_shape,
        splits=(1 - parser.parse_args().validation_split, parser.parse_args().validation_split),
        seed=parser.parse_args().seed,
    )

    class_names, _, _ = get_dataset_stats(num_classes=parser.parse_args().num_classes)

    print("Loading encoder...")

    path_to_weights = os.path.join("./scampi_unsupervised/frameworks/simclr/", parser.parse_args().file_name)
    
    encoder = tf.keras.applications.Xception(
        weights=path_to_weights,
        include_top=False, 
        pooling="avg", 
        input_shape=parser.parse_args().image_shape, 
        )
    model = add_rescaling_layer(encoder)

    model_id = parser.parse_args().file_name.split(".")[0]

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(
        project="scampi",
        name=parser.parse_args().method + "_evaluation_" + model_id, 
        config=vars(parser.parse_args()),
        tags=["evaluation"],
    )

    table = pd.DataFrame()
    table["class"] = class_names
    summary_table = pd.DataFrame()
    summary_table.index = [model_id]
    summary_table["model_id"] = model_id

    print("Extracting features...")

    X_tr = model.predict(x_tr)
    X_te = model.predict(x_te)

    print("Linear evaluation...")

    if parser.parse_args().method == "logistic":
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
        
        summary_table.loc[model_id, "weighted_log_loss"] = log_loss(
            y_te, log_model.predict_proba(X_te), sample_weight=sample_weights
        )
        
        summary_table.loc[model_id, "log_loss"] = log_loss(
            y_te, log_model.predict_proba(X_te)
        )

        # ...the same for the training set
        y_pred_train = log_model.predict(X_tr)

        summary_table.loc[model_id, "weighted_log_loss_train"] = log_loss(
            y_tr,
            log_model.predict_proba(X_tr),
            sample_weight=np.array([class_weights[y] for y in y_tr]),
        )

        summary_table.loc[model_id, "log_loss_train"] = log_loss(
            y_tr, log_model.predict_proba(X_tr)
        )

    elif parser.parse_args().method == "knn":
        # train a k-nearest neighbors model
        k = 2
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)
        y_pred_train = knn.predict(X_tr)

    # compute the accuracy and balanced accuracy and mean precision on the test set
    summary_table.loc[model_id, "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
    summary_table.loc[model_id, "accuracy"] = accuracy_score(y_te, y_pred)
    summary_table.loc[model_id, "mean_precision"] = precision_score(y_te, y_pred, average="macro")

    # ...the same for the training set
    summary_table.loc[model_id, "balanced_accuracy_train"] = balanced_accuracy_score(y_tr, y_pred_train)
    summary_table.loc[model_id, "accuracy_train"] = accuracy_score(y_tr, y_pred_train)
    summary_table.loc[model_id, "mean_precision_train"] = precision_score(y_tr, y_pred_train, average="macro")

    # add the metrics to the table
    table[model_id + "_precision"] = precision_score(y_te, y_pred, average=None)
    table[model_id + "_recall"] = recall_score(y_te, y_pred, average=None)

    # log confusion matrix
    conf_mat = pd.DataFrame(columns=class_names, index=class_names)
    conf_mat.iloc[:, :] = tf.math.confusion_matrix(
        y_te, y_pred, num_classes=parser.parse_args().num_classes
    ).numpy()

    wandb.log({model_id + "_confusion_matrix": wandb.Table(dataframe=conf_mat)})
    wandb.log({"table": wandb.Table(dataframe=table)})
    wandb.log({"summary_table": wandb.Table(dataframe=summary_table)})

    wandb.finish()
