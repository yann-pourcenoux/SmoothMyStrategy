"""Module that contains the class that will hold the data for environments."""

import pandas as pd

import data.loader
import data.preprocessing
from config.schemas import DataLoaderConfigSchema, DataPreprocessingConfigSchema


class DataContainer:
    """Class to hold the data for the environments."""

    data: pd.DataFrame
    num_tickers: int
    num_time_steps: int

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
        self.num_time_steps = max(self.data.index) + 1
