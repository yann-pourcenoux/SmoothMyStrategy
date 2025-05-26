"""Module to hold the config for the data."""

from dataclasses import field

import pydantic


@pydantic.dataclasses.dataclass
class DataPreprocessingConfigSchema:
    """Configuration for DataPreprocessing.

    Attributes:
        technical_indicators (list[str]): list of technical indicators to use.
        start_date (str): start date to use for the data.
        end_date (str): end date to use for the data.
    """

    technical_indicators: list[str] = field(default_factory=list)
    start_date: str = field(default_factory=str)
    end_date: str = field(default_factory=str)


@pydantic.dataclasses.dataclass
class DataLoaderConfigSchema:
    """Configuration for DataLoader.

    Attributes:
        tickers (list[str]): list of tickers to load data for.
    """

    tickers: list[str] = field(default_factory=list)

    @pydantic.field_validator("tickers")
    @classmethod
    def sort_tickers(cls, tickers: list[str]) -> list[str]:
        """Sort the tickers."""
        return sorted(tickers)
