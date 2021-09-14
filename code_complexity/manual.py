import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import scipy.stats


NPERSEG = 60                    # length of segment
NOVERLAP = int(0.75 * NPERSEG)  # overlap of segments
NFFT = NPERSEG                  # length of FFT
WINDOW = "hann"                 # window function type

# code: 9, comments: 0, f./cls. def.: 0, f. calls: 0

def proportion_of_zeros(x: np.ndarray) -> float:
    """
    Calculates proportion of zeros in given array, i.e. number of zeros divided
    by length of array.

    :param x: 1D Numpy array
    :returns: proportion of zeros
    """
    # we may be dealing with floating numbers, we can't use direct comparison
    zeros_count = np.sum(np.isclose(x, 0))
    return zeros_count / len(x)


def power_spectral_density(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates power spectral density (PSD) from "activity" column of a
    DataFrame.

    :param df: DataFrame with "activity" column
    :returns: 1D Numpy array with power spectral density
    """
    psd = scipy.signal.welch(
        x=df["activity"].values, fs=(1 / 60), nperseg=NPERSEG,
        noverlap=NOVERLAP, nfft=NFFT, window=WINDOW, scaling="density"
    )[1]
    return psd


def spectral_flatness(df: pd.DataFrame) -> float:
    """
    Calculates spectral flatness of a signal, i.e. a geometric mean of the
    power spectrum divided by the arithmetic mean of the power spectrum.

    If some frequency bins in the power spectrum are close to zero, they are
    removed prior to calculation of spectral flatness to avoid calculation of
    log(0).

    :param df: DataFrame with "activity" column
    :returns: spectral flatness value
    """
    power_spectrum = scipy.signal.welch(
        x=df["activity"].values, fs=(1 / 60), nperseg=NPERSEG,
        noverlap=NOVERLAP, nfft=NFFT, window=WINDOW, scaling="spectrum"
    )[1]

    non_zeros_mask = ~np.isclose(power_spectrum, 0)
    power_spectrum = power_spectrum[non_zeros_mask]

    return scipy.stats.gmean(power_spectrum) / power_spectrum.mean()


# code: 17, comments: 15, f./cls. def.: 3, f. calls: 4

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from activity signal in time domain.

    :param df_resampled: DataFrame with "activity" column
    :returns: DataFrame with a single row representing features
    """
    X = df["activity"].values

    features = {
        "minimum": np.min(X),
        "maximum": np.max(X),
        "mean": np.mean(X),
        "median": np.median(X),
        "variance": np.var(X, ddof=1),  # apply Bessel's correction
        "kurtosis": sp.stats.kurtosis(X),
        "skewness": sp.stats.skew(X),
        "coeff_of_var": sp.stats.variation(X),
        "iqr": sp.stats.iqr(X),
        "trimmed_mean": sp.stats.trim_mean(X, proportiontocut=0.1),
        "entropy": sp.stats.entropy(X, base=2),
        "proportion_of_zeros": proportion_of_zeros(X)
    }

    return pd.DataFrame([features])

# code: 17, comments: 3, f./cls. def.: 1, f. calls: 12


def extract_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from activity signal in frequency domain, i.e. calculated
    from its Power Spectral Density (PSD).

    :param df: DataFrame with "activity" column
    :returns: DataFrame with a single row representing features
    """
    X = power_spectral_density(df)

    features = {
        "minimum": np.min(X),
        "maximum": np.max(X),
        "mean": np.mean(X),
        "median": np.median(X),
        "variance": np.var(X),
        "kurtosis": sp.stats.kurtosis(X),
        "skewness": sp.stats.skew(X),
        "coeff_of_var": sp.stats.variation(X),
        "iqr": sp.stats.iqr(X),
        "trimmed_mean": sp.stats.trim_mean(X, proportiontocut=0.1),
        "entropy": sp.stats.entropy(X, base=2),
        "spectral_flatness": spectral_flatness(df)
    }

    return pd.DataFrame([features])

# code: 17, comments: 4, f./cls. def.: 1, f. calls: 13


# NOTE: the code below is originally part of extract_features_for_dataframes()
# function; only those lines were specific for manual feature engineering

time_features = extract_time_features(df)
freq_features = extract_frequency_features(df)

merged_features = pd.merge(
    left=time_features, right=freq_features,
    left_index=True, right_index=True, suffixes=["_time", "_freq"]
)
features.append(merged_features)

datasets[part] = pd.concat(features)
datasets[part].reset_index(drop=True, inplace=True)

# code: 8, comments: 0, f./cls. def.: 1, f. calls: 6

