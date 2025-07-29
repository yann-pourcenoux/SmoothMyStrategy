"""Module that contains tests for the weight action environment."""

import unittest

import torch

from environment.test.utils import BaseTestEnvironment, create_env
from environment.weight_action import WeightActionEnv


class TestWeightActionEnv(BaseTestEnvironment, unittest.TestCase):
    """Class to test the WeightAction environment."""

    environment_class = WeightActionEnv

    def test_action_one_ticker(self):
        """Test when a action has to be performed on an env with one ticker and
        artificial weights."""
        batch_size = 2
        fake_divider = 2
        env = create_env(self.environment_class, batch_size=batch_size, tickers=["AAPL"])

        tensordict = env.reset()
        action = torch.ones((batch_size, 1), dtype=torch.float32, device=env.device)
        action = action / fake_divider  # To simulate the weight in a portfolio allocation strategy
        tensordict["action"] = action

        price_day_0 = env.states_per_day[0]["adj_close"]
        num_shares_owned_day_0 = tensordict["num_shares_owned"]
        cash_day_0 = tensordict["cash"]
        value_day_0 = cash_day_0 + torch.sum(
            num_shares_owned_day_0 * price_day_0, dim=-1, keepdim=True
        )

        cash_day_1 = cash_day_0 / fake_divider
        num_shares_owned_day_1 = num_shares_owned_day_0 + cash_day_0 / fake_divider / price_day_0
        price_day_1 = env.states_per_day[1]["adj_close"]
        value_day_1 = cash_day_1 + torch.sum(
            num_shares_owned_day_1 * price_day_1, dim=-1, keepdim=True
        )

        reward = torch.log(value_day_1 / value_day_0) - torch.log(price_day_1 / price_day_0)

        tensordict = env.step(tensordict)

        self.assertTrue(torch.all(tensordict["next"]["reward"] == reward))
        self.assertTrue(torch.all(tensordict["next"]["cash"] == cash_day_1))
        self.assertTrue(torch.all(tensordict["next"]["num_shares_owned"] == num_shares_owned_day_1))
