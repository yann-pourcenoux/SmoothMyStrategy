"""Module that contains the class that will hold the data for environments."""

import pandas as pd

import data.loader
import data.preprocessing
from config.data import DataLoaderConfigSchema, DataPreprocessingConfigSchema


class DataContainer:
    """Class to hold the data for the environments."""

    data: pd.DataFrame
    num_tickers: int

    def __init__(
        self,
        loading_config: DataLoaderConfigSchema,
        preprocessing_config: DataPreprocessingConfigSchema,
    ):
        self._loading_config = loading_config
        self._preprocessing_config = preprocessing_config

        self.data = data.preprocessing.preprocess_data(
            stock_df_iterator=data.loader.load_data(self._loading_config),
            config=self._preprocessing_config,
        )
        self.num_tickers = len(self._loading_config.tickers)

    def _select_time_range(self, start_date: str | None = None, end_date: str | None = None):
        """Select the time range for the environment."""
        df = data.preprocessing.select_time_range(self.data, start_date, end_date)
        df.index = df.date.factorize()[0]
        return df

    def get_env_data(self, start_date: str | None = None, end_date: str | None = None):
        """Get the data for the environment."""
        df = self._select_time_range(start_date, end_date)
        dates = df.date.unique()
        tickers = df.ticker.unique()

        df.drop(columns=["date", "ticker"], inplace=True)
        num_time_steps = max(df.index)

        return df, num_time_steps, dates, tickers
