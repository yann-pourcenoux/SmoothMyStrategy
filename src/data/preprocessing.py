"""Data preprocessing module."""

from typing import Iterable

import pandas as pd
import pydantic
import stockstats


class DataPreprocessingConfigSchema(pydantic.BaseModel):
    """Configuration for DataPreprocessing.

    Attributes:
        technical_indicators (list[str]): list of technical indicators to use.
    """

    technical_indicators: list[str] = ["macd", "boll_30", "boll_60"]
    start_date: str | None = None
    end_date: str | None = None
    # TODO: add vix and turbulence


def preprocess_data(
    stock_df_iterator: Iterable[stockstats.StockDataFrame],
    config: DataPreprocessingConfigSchema,
) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        stock_df_iterator (Iterable[stockstats.StockDataFrame]): Iterable of
            stockstats.StockDataFrame.
        config (DataPreprocessingConfigSchema): Configuration for preprocessing.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    dataframe_iterator = _add_technical_indicators(
        stock_df_iterator, config.technical_indicators
    )

    dataframe = merge_dataframes(dataframe_iterator)

    dataframe = select_time_range(dataframe, config.start_date, config.end_date)
    dataframe = clean_data(dataframe)

    # Drop the date and ticker columns
    dataframe.drop(columns=["date", "ticker"], inplace=True)

    return dataframe


def _add_technical_indicators(
    stock_df_iterator: Iterable[stockstats.StockDataFrame],
    technical_indicators: list[str],
) -> Iterable[pd.DataFrame]:
    """Add technical indicators to dataframe.

    Args:
        stock_df_iterator (Iterable[stockstats.StockDataFrame]): Iterable of
            stockstats.StockDataFrame.
        technical_indicators (list[str]): list of technical indicators to add.

    Returns:
        Iterable[pd.DataFrame]: Iterable of pd.DataFrame.
    """
    for stock_df in stock_df_iterator:
        for indicator in technical_indicators:
            stock_df.get(indicator)

        df = stockstats.unwrap(stock_df)
        df.reset_index(inplace=True)
        df.dropna(inplace=True)

        yield df


def merge_dataframes(dataframe_iterator: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge dataframes.

    Args:
        dataframe_iterator (Iterable[pd.DataFrame]): Iterable of pd.DataFrame.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    dataframe = pd.concat(dataframe_iterator, ignore_index=True)
    dataframe = dataframe.sort_values(["date", "ticker"], ignore_index=True)
    dataframe.reset_index(drop=True, inplace=True)

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
        dataframe = dataframe[dataframe.date >= pd.to_datetime(start).date()]

    if end is not None:
        dataframe = dataframe[dataframe.date <= pd.to_datetime(end).date()]

    return dataframe


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the data.

    Args:
        dataframe (pd.DataFrame): Dataframe to clean.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = dataframe.copy()
    df.sort_values(["date", "ticker"], ignore_index=True, inplace=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="ticker", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.ticker.isin(tics)]

    return df
