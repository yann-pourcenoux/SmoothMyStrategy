"""Data Loader module."""

import logging
import os
from dataclasses import dataclass

import pandas as pd

from data.constants import FINANCE_DATA_PATH

LOGGER = logging.getLogger(__name__)


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
                os.path.join(FINANCE_DATA_PATH, f"{ticker}.csv"),
                index_col="Date",
                date_format="%Y-%m-%d",
            )
            data[ticker].index = data[ticker].index.map(
                lambda x: pd.to_datetime(x).date()
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
        fillna: bool = True,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """Get dataframe of loaded data.

        Args:
            ticker (str | list[str]): ticker(s) to get dataframe for.
            columns (str | (list[str]): column(s) to get dataframe for.
            start (str | None, optional): start date of dataframe. Defaults to None.
            end (str | None, optional): max date of dataframe. Defaults to None.
            resampling_frequency (str | None, optional): resampling frequency of
                dataframe. Defaults to None.
            fillna (bool, optional): whether to fill NaN values. Defaults to True.
            dropna (bool, optional): whether to drop columns that have NaN values.
                Defaults to False.

        Returns:
            pd.DataFrame: dataframe of loaded data for a ticker.
        """
        if dropna and not fillna:
            LOGGER.warning("It is risky to drop NaN values without filling them.")

        dataframe = select_tickers_columns(self._dataframes, tickers, columns)

        dataframe = select_time_range(dataframe, start, end)

        if fillna:
            dataframe = fill_nan(dataframe)
            columns_with_nan = _get_columns_with_nan(dataframe)
            LOGGER.warning(
                'Could not fill NaN values for columns: "%s"', columns_with_nan
            )

        if dropna:
            columns_with_nan = _get_columns_with_nan(dataframe)
            LOGGER.info('Dropping columns: "%s"', columns_with_nan)
            dataframe.dropna(axis=1, inplace=True)

        if resampling_frequency is not None:
            dataframe = dataframe.resample(resampling_frequency).last()

        return dataframe


def _get_columns_with_nan(dataframe: pd.DataFrame) -> list[str]:
    """Get columns with NaN values.

    Args:
        dataframe (pd.DataFrame): dataframe to get columns with NaN values for.

    Returns:
        list[str]: list of columns with NaN values.
    """
    return dataframe.columns[dataframe.isna().any()].tolist()


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
        dataframe = dataframe[dataframe.index >= pd.to_datetime(start).date()]

    if end is not None:
        dataframe = dataframe[dataframe.index <= pd.to_datetime(end).date()]

    return dataframe


def fill_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to fill NaN values for.

    Returns:
        pd.DataFrame: dataframe with filled NaN values.
    """
    # Fill the "holes" with the last known value
    dataframe.ffill(inplace=True)
    # If there are remaining NaN values, there are at the beginning,
    # fill with next known value
    dataframe.bfill(inplace=True)
    return dataframe
