import os
from typing import Dict, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ArrayLike = TypeVar("ArrayLike", bound=Union[np.ndarray, pd.DataFrame])


class Dataset:
    def __init__(self, dirpath: str, condition_dir_name: str = "condition"):
        condition_dirpath = os.path.join(dirpath, condition_dir_name)
        control_dirpath = os.path.join(dirpath, "control")

        self.condition: List[pd.DataFrame] = [
            pd.read_csv(os.path.join(condition_dirpath, file))
            for file in os.listdir(condition_dirpath)
        ]

        self.control: List[pd.DataFrame] = [
            pd.read_csv(os.path.join(control_dirpath, file))
            for file in os.listdir(control_dirpath)
        ]


def variance_thresholding(
    X_train: ArrayLike,
    X_test: ArrayLike,
    threshold: float
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Filters out those features from data that have variance lower than
    threshold. Variance is calculated on training data only (first scaled to
    range [0, 1] to enable direct comparison of variances), and then resulting
    filtering is applied to test data.

    :param X_train: training data
    :param X_test: test data
    :param threshold: variance threshold, features with variance lower than
    this will be rejected
    :returns: tuple of transformed (X_train, X_test)
    """
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    X_train_scaled = scaler.fit_transform(X_train)

    thresholder = VarianceThreshold(threshold=threshold)
    thresholder.fit(X_train_scaled)

    if isinstance(X_train, np.ndarray):
        X_train = thresholder.transform(X_train)
    elif isinstance(X_train, pd.DataFrame):
        X_train = X_train.loc[:, thresholder.variances_ >= threshold]
    else:
        raise ValueError(f"X_train should be Numpy array or Pandas DataFrame, "
                         f"got: {type(X_train)}")

    if isinstance(X_test, np.ndarray):
        X_test = thresholder.transform(X_test)
    elif isinstance(X_test, pd.DataFrame):
        X_test = X_test.loc[:, thresholder.variances_ >= threshold]
    else:
        raise ValueError(f"X_test should be Numpy array or Pandas DataFrame, "
                         f"got: {type(X_test)}")

    return X_train, X_test


def standardize(
    X_train: ArrayLike,
    X_test: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Performs standardization, i.e. subtract mean and divide by standard deviation for each feature.
    Calculates mean and standard deviation using only training data.

    :param X_train: training data
    :param X_test: test data
    :returns: tuple of transformed (X_train, X_test)
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    if isinstance(X_train, np.ndarray):
        X_train = scaler.transform(X_train)
    elif isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(
            data=scaler.transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
    else:
        raise ValueError(f"X_train should be Numpy array or Pandas DataFrame, "
                         f"got: {type(X_train)}")

    if isinstance(X_test, np.ndarray):
        X_test = scaler.transform(X_test)
    elif isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(
            data=scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
    else:
        raise ValueError(f"X_test should be Numpy array or Pandas DataFrame, "
                         f"got: {type(X_test)}")

    return X_train, X_test


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Matthews Correlation Coefficient (MCC) using the definion based directly on confusion matrix.

    If denominator is 0, it is set to 1 to avoid division by zero.

    If there is only one sample, 1 is returned in case of accurate prediction and 0 otherwise.

    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :returns: Matthews Correlation Coefficient (MCC)
    """
    if len(y_true) == 1:
        return y_true == y_pred

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if np.isclose(denominator, 0):
        denominator = 1

    return numerator / denominator


def calculate_metrics(clf: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Calculates metrics on test set for fitted classifier.

    :param clf: fitted Scikit-learn compatibile classifier
    :param X_test: test data
    :param y_test: true test labels
    :returns: dictionary: metric name -> metric value
    """
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=1),
        "precision": precision_score(y_test, y_pred, zero_division=1),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=1),
        "specificity": recall_score(y_test, y_pred, pos_label=0, zero_division=1),
        "ROC_AUC": roc_auc_score(y_test, y_pred),
        "MCC": mcc(y_test, y_pred)
    }

    return metrics


def calculate_metrics_statistics(metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    For list of dicts, each containing metric name -> metric value (same metrics), calculates mean and
    standard deviation for each metric.

    :param metrics: list of dictionaries of metrics, all with the same keys
    :returns: dictionary: metric name -> (mean metric value, std dev of metric)
    """
    results = {}
    metrics_names = metrics[0].keys()

    for metric in metrics_names:
        values = [fold_metrics[metric] for fold_metrics in metrics]
        mean = np.mean(values)
        stddev = np.std(values)
        results[metric] = mean, stddev

    return results
