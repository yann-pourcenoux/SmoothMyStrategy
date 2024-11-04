"""Module that contains the functions to add features to the data."""

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd


def get_feature(
    feature_name: str, feature_params: dict[str, Any]
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Get a feature from the feature name."""
    if feature_name.startswith("log_return"):
        return partial(log_return, **feature_params)
    elif feature_name.startswith("macd"):
        return partial(macd, **feature_params)
    else:
        raise ValueError(f"Feature {feature_name} not found")


def log_return(
    dataframe: pd.DataFrame,
    column: str = "adj_close",
    shift: int = 0,
) -> pd.DataFrame:
    """Add the log return to the dataframe."""
    dataframe[f"log_return_{shift}"] = np.log(
        dataframe[column] / dataframe[column].shift(1)
    )
    dataframe[f"log_return_{shift}"] = dataframe[f"log_return_{shift}"].shift(shift)
    return dataframe


def macd(
    dataframe: pd.DataFrame,
    column: str = "adj_close",
    shift: int = 26,
) -> pd.DataFrame:
    ema_12 = dataframe[column].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe[column].ewm(span=26, adjust=False).mean()

    ema_12 = ema_12 / dataframe[column].shift(shift)
    ema_26 = ema_26 / dataframe[column].shift(shift)

    dataframe[f"macd_{shift}"] = ema_12 - ema_26
    return dataframe
