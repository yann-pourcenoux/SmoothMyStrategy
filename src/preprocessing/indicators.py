"""Module that contains indicators."""

import pandas as pd

import preprocessing.moving_average
import preprocessing.volatility


def bollinger_bands(
    dataframe: pd.DataFrame,
    time_window: int = 20,
    moving_average: str = "simple",
    stddev_coef: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bolinger bands for dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to compute bolinger bands for.
        time_window (int, optional): time window for bolinger bands. Defaults to 20.
        moving_average (str, optional): moving average to use. Defaults to "simple".
        stddev_coef (float, optional): coefficient for standard deviation.
            Defaults to 2.0.

    Returns:
        pd.DataFrame: dataframe with lower band.
        pd.DataFrame: dataframe with upper band.
    """
    if moving_average not in ["simple", "exponential"]:
        raise ValueError("moving_average must be 'simple' or 'exponential'")

    moving_average_fn = preprocessing.moving_average.simple_moving_average
    if moving_average == "exponential":
        moving_average_fn = preprocessing.moving_average.exponential_moving_average

    mean = moving_average_fn(dataframe, time_window)
    stddev = preprocessing.volatility.standard_deviation(dataframe, time_window)

    upper_band = mean + stddev_coef * stddev
    lower_band = mean - stddev_coef * stddev

    return lower_band, upper_band
