from copy import deepcopy
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.transformers import FeatureAugmenter


# code: 6, comments: 0, , f./cls. def.: 0, f. calls: 0


def get_tsfresh_flat_format_df(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Creates DataFrame in a "flat" format for tsfresh from list of DataFrames.
    Each one is assumed to have "timestamp" and "activity" columns.

    :param dfs: list of DataFrames; each one has to have "timestamp" and
    "activity" columns
    :returns: DataFrame in tsfresh "flat" format
    """
    dfs = deepcopy(dfs)  # create copy to avoid side effects

    flat_df = pd.DataFrame(columns=["id", "timestamp", "activity"])

    for idx, df in enumerate(dfs):
        df["id"] = idx
        flat_df = flat_df.append(df)

    flat_df = flat_df.reset_index(drop=True)

    return flat_df

# code: 8, comments: 5, f./cls. def.: 1, f. calls: 6


def extract_tsfresh_features(dfs: List[pd.DataFrame], settings: Dict) \
        -> pd.DataFrame:
    """
    Performs feature extraction (only extraction, not selection) using tsfresh.

    :param dfs: list of DataFrames with time series, each with "timestamp" and
    "activity" columns
    :param settings: tsfresh settings, one of: ComprehensiveFCParameters,
    EfficientFCParameters, MinimalFCParameters
    :returns: DataFrame with extracted features, with one row per original
    DataFrame with time series (in the same order)
    """
    ts = get_tsfresh_flat_format_df(dfs)
    ids = ts["id"].unique()
    X = pd.DataFrame(index=ids)

    augmenter = FeatureAugmenter(
        default_fc_parameters=settings, column_id="id",
        column_sort="timestamp", column_value="activity", chunksize=1, n_jobs=4
    )

    augmenter.set_timeseries_container(ts)
    X = augmenter.transform(X)

    return X


# code: 12, comments: 7, f./cls. def.: 1, f. calls: 6


class TsfreshTopNFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects top N features using tsfresh feature selector.
    """

    def __init__(self, n: int = 10):
        self.n: int = n
        self.features: List[int] = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        relevance_table = calculate_relevance_table(X, y)
        relevance_table.sort_values("p_value", inplace=True)
        features = relevance_table.head(self.n)["feature"]
        self.features = list(features.values)

    def transform(self, X, y=None):
        return X[:, self.features]


# code: 18, comments: 1, f./cls. def.: 1, f. calls: 9

# NOTE: the code below is originally part of feature extraction; only this
# line was specific for automated feature engineering
X = extract_tsfresh_features(dfs, settings)

# code: 1, comments: 0, f./cls. def.: 0, f. calls: 1

# NOTE: the code below is originally part of preprocessing directly before
# training classifiers; only those lines were specific for automated feature
# engineering

selector = IncreasingFDRFeatureSelector(verbose=True)
selector.fit(X_train, y_train)
X_train, X_test = selector.transform(X_train), selector.transform(X_test)

# code: 3, comments: 0, f./cls. def.: 0, f. calls: 4
