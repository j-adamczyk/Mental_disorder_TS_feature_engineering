from copy import deepcopy
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.transformers import FeatureAugmenter, FeatureSelector


# code: 5, comments: 0, f./cls. def.: 0, f. calls: 0


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

# code: 8, comments: 5, f./cls. def.: 1, f. calls: 5


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


class IncreasingFDRFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Class for selecting features based on tsfresh feature selector. FDR starts
    at 0.05 and is increased if no features are selected at that level.
    """

    def __init__(self, verbose: bool = False):
        self.selector: FeatureSelector = None
        self.verbose: bool = verbose

    def fit(self, X, y):
        final_alpha = None
        for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            self.selector = FeatureSelector(fdr_level=alpha, n_jobs=4,
                                            chunksize=1)
            self.selector.fit(X, y)
            if len(self.selector.relevant_features) > 0:
                if self.verbose:
                    print("FDR:", final_alpha)
                return selector

        raise ValueError("Failed to select any features")

    def transform(self, X):
        return self.selector.transform(X)


# code: 18, comments: 2, f./cls. def.: 1, f. calls: 5

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
