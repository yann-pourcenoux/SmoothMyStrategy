"""Module to compute volatility."""


import pandas as pd


def standard_deviation(dataframe: pd.DataFrame, time_window: int) -> pd.DataFrame:
    """Compute standard deviation for dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to compute standard deviation for.
        time_window (int): time window for standard deviation.

    Returns:
        pd.DataFrame: dataframe with standard deviation.
    """
    return dataframe.rolling(time_window).std()
