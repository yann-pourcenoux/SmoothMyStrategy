"""Module that contains tests for the trading environment."""

import abc

from torchrl.envs.utils import check_env_specs

from config.data import DataLoaderConfigSchema, DataPreprocessingConfigSchema
from config.environment import EnvironmentConfigSchema
from data.container import DataContainer
from environment.base import BaseTradingEnv
from environment.utils import apply_transforms

START_DATE = "2020-01-01"
END_DATE = "2024-01-01"


def create_env(
    environment_class: BaseTradingEnv,
    batch_size: int | None,
    tickers: list[str],
) -> BaseTradingEnv:
    env = environment_class(
        config=EnvironmentConfigSchema(
            batch_size=batch_size,
            start_date=START_DATE,
            end_date=END_DATE,
        ),
        data_container=DataContainer(
            loading_config=DataLoaderConfigSchema(tickers=tickers),
            preprocessing_config=DataPreprocessingConfigSchema(
                technical_indicators=[
                    "log_return_1",
                ],
                start_date=START_DATE,
                end_date=END_DATE,
            ),
        ),
    )
    return env


def check_env_bs_tickers(
    environment_class: BaseTradingEnv,
    batch_size: int | None,
    tickers: list[str],
):
    """Check the environment from the configs."""
    env = create_env(environment_class, batch_size, tickers)
    check_env_specs(env)
    env = apply_transforms(env)
    check_env_specs(env)


class BaseTestEnvironment(abc.ABC):
    """Base class to test the trading environment."""

    environment_class: BaseTradingEnv

    def test_one_batch_one_ticker(self):
        """Test the trading environment with a batch size of 1 and one ticker."""
        check_env_bs_tickers(
            self.environment_class,
            batch_size=1,
            tickers=["AAPL"],
        )

    def test_one_batch_multiple_tickers(self):
        """Test the trading environment with a batch size of 1 and multiple tickers."""
        check_env_bs_tickers(
            self.environment_class,
            batch_size=1,
            tickers=["AAPL", "MSFT", "GOOGL"],
        )

    def test_batch_one_ticker(self):
        """Test the trading environment with batch and one ticker."""
        check_env_bs_tickers(
            self.environment_class,
            batch_size=2,
            tickers=["AAPL"],
        )

    def test_batch_multiple_tickers(self):
        """Test the trading environment with batch and multiple tickers."""
        check_env_bs_tickers(
            self.environment_class,
            batch_size=2,
            tickers=["AAPL", "MSFT", "GOOGL"],
        )
