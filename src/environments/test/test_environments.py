"""Module that contains tests for the trading environment."""

import abc
import unittest

import torch
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

from common.config import DataLoaderConfigSchema, DataPreprocessingConfigSchema
from data.container import DataContainer
from environments.config import EnvironmentConfigSchema
from environments.trading import TradingEnv, apply_transforms

START_DATE = "2020-01-01"
END_DATE = "2024-01-01"


def _create_env(
    environment_class: EnvBase,
    batch_size: int | None,
    tickers: list[str],
) -> TradingEnv:
    env = environment_class(
        config=EnvironmentConfigSchema(
            batch_size=batch_size,
            start_date=START_DATE,
            end_date=END_DATE,
        ),
        data_container=DataContainer(
            loading_config=DataLoaderConfigSchema(tickers=tickers),
            preprocessing_config=DataPreprocessingConfigSchema(
                technical_indicators=["close_10_sma", "log-ret"],
                start_date=START_DATE,
                end_date=END_DATE,
            ),
        ),
    )
    return env


def _check_env_bs_tickers(
    environment_class: EnvBase,
    batch_size: int | None,
    tickers: list[str],
):
    """Check the environment from the configs."""
    env = _create_env(environment_class, batch_size, tickers)
    check_env_specs(env)
    env = apply_transforms(env)
    check_env_specs(env)


class BaseTestTradingEnvironment(abc.ABC):
    """Base class to test the trading environment."""

    environment_class: EnvBase

    def test_one_batch_one_ticker(self):
        """Test the trading environment with a batch size of 1 and one ticker."""
        _check_env_bs_tickers(
            self.environment_class,
            batch_size=1,
            tickers=["AAPL"],
        )

    def test_one_batch_multiple_tickers(self):
        """Test the trading environment with a batch size of 1 and multiple tickers."""
        _check_env_bs_tickers(
            self.environment_class,
            batch_size=1,
            tickers=["AAPL", "MSFT", "GOOGL"],
        )

    def test_batch_one_ticker(self):
        """Test the trading environment with batch and one ticker."""
        _check_env_bs_tickers(
            self.environment_class,
            batch_size=2,
            tickers=["AAPL"],
        )

    def test_batch_multiple_tickers(self):
        """Test the trading environment with batch and multiple tickers."""
        _check_env_bs_tickers(
            self.environment_class,
            batch_size=2,
            tickers=["AAPL", "MSFT", "GOOGL"],
        )


class TestTradingEnv(BaseTestTradingEnvironment, unittest.TestCase):
    """Class to test the base trading environment."""

    environment_class = TradingEnv

    def test_action_one_ticker(self):
        """Test when a action has to be performed on an env with only one ticker."""
        batch_size = 2
        env = _create_env(
            self.environment_class, batch_size=batch_size, tickers=["AAPL"]
        )

        tensordict = env.reset()
        action = torch.ones((batch_size, 1), dtype=torch.float32, device=env.device)
        tensordict["action"] = action

        price_day_0 = env.states_per_day[0]["close"]
        num_shares_owned_day_0 = tensordict["num_shares_owned"]
        cash_day_0 = tensordict["cash"]
        assert torch.all(cash_day_0 > action * price_day_0)
        value_day_0 = cash_day_0 + torch.sum(
            num_shares_owned_day_0 * price_day_0, dim=-1, keepdim=True
        )

        cash_day_1 = cash_day_0 - price_day_0 * action
        num_shares_owned_day_1 = num_shares_owned_day_0 + action
        price_day_1 = env.states_per_day[1]["close"]
        value_day_1 = cash_day_1 + torch.sum(
            num_shares_owned_day_1 * price_day_1, dim=-1, keepdim=True
        )

        reward = torch.log(value_day_1 / value_day_0)

        tensordict = env._perform_trading_action(tensordict)

        self.assertTrue(torch.all(tensordict["reward"] == reward))
        self.assertTrue(torch.all(tensordict["cash"] == cash_day_1))
        self.assertTrue(
            torch.all(tensordict["num_shares_owned"] == num_shares_owned_day_1)
        )


if __name__ == "__main__":
    unittest.main()
