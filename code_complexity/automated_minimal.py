from copy import deepcopy
from typing import Dict, List

import pandas as pd
from tsfresh.transformers import FeatureAugmenter

# code: 4, comments: 0, , f./cls. def.: 0, f. calls: 0


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
        default_fc_parameters=settings,
        column_id="id",
        column_sort="timestamp",
        column_value="activity",
        chunksize=1,
        n_jobs=4
    )

    augmenter.set_timeseries_container(ts)
    X = augmenter.transform(X)

    return X

# code: 12, comments: 7, f./cls. def.: 1, f. calls: 6

# NOTE: the code below is originally part of feature extraction; only this
# line was specific for automated feature engineering
X = extract_tsfresh_features(dfs, settings)

# code: 1, comments: 0, f./cls. def.: 0, f. calls: 1

