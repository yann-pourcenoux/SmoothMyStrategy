"""Data preprocessing module."""

from typing import Iterable

import pandas as pd

from config.data import DataPreprocessingConfigSchema
from data.features import FeatureGenerator


def preprocess_data(
    stock_df_iterator: Iterable[pd.DataFrame],
    config: DataPreprocessingConfigSchema,
) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        stock_df_iterator (Iterable[pd.DataFrame]): Iterable of pd.DataFrame.
        config (DataPreprocessingConfigSchema): Configuration for preprocessing.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    dataframe_iterator = _add_technical_indicators(stock_df_iterator, config.technical_indicators)

    dataframe = merge_dataframes(dataframe_iterator)
    dataframe = _add_date_features(dataframe)

    dataframe = select_time_range(dataframe, config.start_date, config.end_date)
    dataframe = clean_data(dataframe)

    return dataframe


def _add_technical_indicators(
    stock_df_iterator: Iterable[pd.DataFrame],
    technical_indicators: list[str],
) -> Iterable[pd.DataFrame]:
    """Add technical indicators to dataframe.

    Args:
        stock_df_iterator (Iterable[pd.DataFrame]): Iterable of pd.DataFrame.
        technical_indicators (dict[str, dict[str, Any]]): dictionary of technical
            indicators to add.

    Returns:
        Iterable[pd.DataFrame]: Iterable of pd.DataFrame.
    """
    feature_generator = FeatureGenerator()
    for stock_df in stock_df_iterator:
        df = feature_generator.generate_features(stock_df, technical_indicators)

        df.reset_index(inplace=True)
        df.dropna(inplace=True)

        yield df


def _add_date_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate the date features.

    Args:
        data: DataFrame containing price data

    Returns:
        DataFrame containing the date features
    """
    dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
    dataframe["day_of_month"] = dataframe["date"].dt.day
    dataframe["day_of_year"] = dataframe["date"].dt.dayofyear
    dataframe["month_of_year"] = dataframe["date"].dt.month
    dataframe["is_first_day_of_month"] = dataframe["month_of_year"] != dataframe[
        "month_of_year"
    ].shift(1)

    return dataframe


def reset_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sort the rows by date and ticker, then reset the index.

    Args:
        dataframe (pd.DataFrame): Dataframe to reset index for.

    Returns:
        pd.DataFrame: Dataframe with reset index.
    """
    dataframe = dataframe.sort_values(["date", "ticker"], ignore_index=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def merge_dataframes(dataframe_iterator: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge dataframes.

    Args:
        dataframe_iterator (Iterable[pd.DataFrame]): Iterable of pd.DataFrame.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    dataframe = pd.concat(dataframe_iterator, ignore_index=True)
    reset_index(dataframe)

    return dataframe


def select_time_range(dataframe: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Select time range of dataframe.

    Args:
        dataframe (pd.DataFrame): dataframe to select time range for.
        start (str | None, optional): start date of dataframe. Defaults to None.
        end (str | None, optional): max date of dataframe. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with selected time range.
    """
    if start is not None:
        dataframe = dataframe[dataframe.date.dt.date >= pd.to_datetime(start).date()]

    if end is not None:
        dataframe = dataframe[dataframe.date.dt.date < pd.to_datetime(end).date()]

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

    # Fix this by not dropping but filling the NaNs value
    # which are probably linked to not having old enough data
    assert set(tics) == set(df.ticker.unique()), "Some tickers have missing values."

    df = df[df.ticker.isin(tics)]

    return df
