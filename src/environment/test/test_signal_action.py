"""Module that contains tests for the trading environment."""

import unittest

import torch

from environment.signal_action import SignalActionEnv
from environment.test.utils import BaseTestEnvironment, create_env


class TestTradingEnv(BaseTestEnvironment, unittest.TestCase):
    """Class to test the base trading environment."""

    environment_class = SignalActionEnv

    def test_action_one_ticker(self):
        """Test when a action has to be performed on an env with only one ticker."""
        batch_size = 2
        env = create_env(self.environment_class, batch_size=batch_size, tickers=["AAPL"])

        tensordict = env.reset()
        action = torch.ones((batch_size, 1), dtype=torch.float32, device=env.device)
        tensordict["action"] = action

        price_day_0 = env.states_per_day[0]["adj_close"]
        num_shares_owned_day_0 = tensordict["num_shares_owned"]
        cash_day_0 = tensordict["cash"]
        assert torch.all(cash_day_0 > action * price_day_0)
        value_day_0 = cash_day_0 + torch.sum(
            num_shares_owned_day_0 * price_day_0, dim=-1, keepdim=True
        )

        cash_day_1 = cash_day_0 - price_day_0 * action
        num_shares_owned_day_1 = num_shares_owned_day_0 + action
        price_day_1 = env.states_per_day[1]["adj_close"]
        value_day_1 = cash_day_1 + torch.sum(
            num_shares_owned_day_1 * price_day_1, dim=-1, keepdim=True
        )

        reward = torch.log(value_day_1 / value_day_0) - torch.log(price_day_1 / price_day_0)

        tensordict = env.step(tensordict)

        self.assertTrue(torch.all(tensordict["next"]["reward"] == reward))
        self.assertTrue(torch.all(tensordict["next"]["cash"] == cash_day_1))
        self.assertTrue(torch.all(tensordict["next"]["num_shares_owned"] == num_shares_owned_day_1))
