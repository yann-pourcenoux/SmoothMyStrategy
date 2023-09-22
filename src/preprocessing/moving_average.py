"""Module to compute moving averages."""

import pandas as pd


def simple_moving_average(dataframe: pd.DataFrame, time_window: int) -> pd.DataFrame:
    """Compute simple moving average of dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to compute simple moving average for.
        time_window (int): time window to compute simple moving average for.

    Returns:
        pd.DataFrame: dataframe with simple moving average.
    """
    return dataframe.rolling(time_window).mean()


def exponential_moving_average(
    dataframe: pd.DataFrame, time_window: int
) -> pd.DataFrame:
    """Compute exponential moving average of dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to compute exponential moving average for.
        time_window (int): time window to compute exponential moving average for.

    Returns:
        pd.DataFrame: dataframe with exponential moving average.
    """
    return dataframe.ewm(span=time_window, adjust=False).mean()
