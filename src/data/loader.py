"""Data Loader module."""

import os
from typing import Iterable

import pandas as pd

from config.data import DataLoaderConfigSchema
from data.constants import DATASET_PATH


def load_data(config: DataLoaderConfigSchema) -> Iterable[pd.DataFrame]:
    """Load data from csv files.

    Args:
        config (DataLoaderConfigSchema): configuration for DataLoader.

    Returns:
        Iterable[pd.DataFrame]: iterable of pd.DataFrame.
    """
    for ticker in config.tickers:
        df = pd.read_csv(os.path.join(DATASET_PATH, f"{ticker}.csv"))
        df = df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
        ]

        df = _select_and_rename_columns(df)
        df = _convert_to_date(df)

        df["ticker"] = ticker
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        yield df


def _select_and_rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns of dataframe inplace.

    Args:
        dataframe (pd.DataFrame): dataframe to rename columns for.

    Returns:
        pd.DataFrame: dataframe with renamed columns.
    """
    mapping = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    columns_to_drop = [column for column in dataframe.columns if column not in mapping]
    dataframe.drop(columns=columns_to_drop, inplace=True)
    dataframe.rename(columns=mapping, inplace=True)
    return dataframe


def _convert_to_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert "Date" column of dataframe to datetime format inplace.

    Args:
        dataframe (pd.DataFrame): dataframe to convert "Date" column for.

    Returns:
        pd.DataFrame: dataframe with converted "Date" column.
    """
    dataframe["date"] = pd.to_datetime(dataframe.date)
    return dataframe
