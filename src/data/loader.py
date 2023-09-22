"""Data Loader module."""

import os
from dataclasses import dataclass

import pandas as pd

from data.constants import FINANCE_DATA_PATH


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader.

    Attributes:
        tickers (list[str]): list of tickers to load data for.
    """

    tickers: list[str]


class DataLoader:
    """Class to load data from csv files."""

    def __init__(self, config: DataLoaderConfig):
        """Initialize DataLoader.

        Args:
            config (DataLoaderConfig): configuration for DataLoader.
        """
        self._config = config
        self._dataframes = self._load_data()

    def _load_data(self) -> dict[str, pd.DataFrame]:
        """Load data from csv files.

        Returns:
            dict[str, pd.DataFrame]: Dictionary of dataframe of loaded data.
        """
        data = {}
        for ticker in self._config.tickers:
            data[ticker] = pd.read_csv(
                os.path.join(FINANCE_DATA_PATH, f"{ticker}.csv"), index_col="Date"
            )
        return data

    def get_dataframe(
        self,
        tickers: str | list[str],
        columns: str | list[str],
        *,
        start: str | None = None,
        end: str | None = None,
        resampling_frequency: str | None = None,
    ) -> pd.DataFrame:
        """Get dataframe of loaded data.

        Args:
            ticker (str | list[str]): ticker(s) to get dataframe for.
            columns (str | (list[str]): column(s) to get dataframe for.
            start (str | None, optional): start date of dataframe. Defaults to None.
            end (str | None, optional): max date of dataframe. Defaults to None.
            resampling_frequency (str | None, optional): resampling frequency of
                dataframe. Defaults to None.

        Returns:
            pd.DataFrame: dataframe of loaded data for a ticker.
        """
        dataframe = select_tickers_columns(self._dataframes, tickers, columns)

        dataframe = select_time_range(dataframe, start, end)

        dataframe = fill_nan(dataframe)

        if resampling_frequency is not None:
            dataframe = dataframe.resample(resampling_frequency).last()

        return dataframe


def select_tickers_columns(
    dataframes: dict[str, pd.DataFrame],
    tickers: str | list[str],
    columns: str | list[str],
) -> pd.DataFrame:
    """Select tickers and columns of dataframes.

    Args:
        dataframes (dict[str, pd.DataFrame]): dataframes to select tickers and columns
            for.
        tickers (str | list[str]): ticker(s) to select.
        columns (str | list[str]): column(s) to select.

    Returns:
        pd.DataFrame: dataframe with selected tickers and columns.
    """
    dataframe = pd.DataFrame()

    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(columns, str):
        columns = [columns]

    for ticker in tickers:
        for column in columns:
            dataframe[f"{ticker}_{column}"] = dataframes[ticker][column]

    return dataframe


def select_time_range(
    dataframe: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    """Select time range of dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to select time range for.
        start (str | None, optional): start date of dataframe. Defaults to None.
        end (str | None, optional): max date of dataframe. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with selected time range.
    """
    if start is not None:
        dataframe = dataframe[dataframe.index >= start]

    if end is not None:
        dataframe = dataframe[dataframe.index <= end]

    return dataframe


def fill_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to fill NaN values for.

    Returns:
        pd.DataFrame: dataframe with filled NaN values.
    """
    # Fill the "holes" with the last known value
    dataframe.fillna(method="ffill", inplace=True)
    # If there are remaining NaN values, there are at the beginning,
    # fill with next known value
    dataframe.fillna(method="bfill", inplace=True)
    return dataframe
