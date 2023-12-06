"""Module to have a Finance stock trading environment.

This is inspired by the implementation of by FinRL on the link below. However, it has
been chosen to be implemented fully using PyTorch directly to be more efficient since
resources can be a bottleneck in this project. Thus we are not using Gym which could
have helped to test the strategies on simpler environments.

https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py#L484
"""


from collections import namedtuple
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import pydantic

State = namedtuple("State", ["observable_state", "portfolio_state", "cash_amount"])


class BrokerFeesConfigSchema(pydantic.BaseModel):
    """Configuration schema for the broker fees.

    Attributes:
        cost_pct (float): percentage of the cost of the transaction.
        min_cost (float): minimum cost of the transaction.
    """

    cost_pct: float = 0.0025
    min_cost: float = 1.0


class EnvironmentConfigSchema(pydantic.BaseModel):
    """Configuration schema for the environment."""

    fees: BrokerFeesConfigSchema = BrokerFeesConfigSchema()
    actions_ticker_list: List[str]
    initial_cash_amount: float = 1e6
    reward_scaling: float = 1e-4
    # TODO: Add turbulence threshold and risk indicator column
    # TODO: add plotting and monitoring graphs
    # TODO: add initial stock portfolio

    def __init__(self, **data: dict) -> None:
        """Initialize the environment configuration."""
        data["actions_ticker_list"] = sorted(data["actions_ticker_list"])
        super().__init__(**data)


class Environment:
    """Stock trading environment."""

    # TODO: consider using a deque instead of a list
    _day: int
    _total_asset_value_memory: List[float]
    _state_memory: List[State]
    _rewards_memory: List[float]
    _performed_actions_memory: List[List[int]]
    _raw_actions_memory: List[List[int]]

    def __init__(
        self, config: EnvironmentConfigSchema, dataframe: pd.DataFrame
    ) -> None:
        """Initialize the environment."""
        self._dataframe = dataframe
        self._config = config
        self._episode: int = 0
        # TODO: add counter of trades

        self._initiate_env()

    def step(self, actions: List[int]) -> Tuple[State, float, bool, dict]:
        """Perform a step of the agent from the actions on the environment.

        Args:
            actions (List[int]): actions to perform on the environment.

        Returns:
            state (State): state of the environment after the actions.
            reward (float): reward of the agent after the actions.
            done (bool): if the episode is finished.
            info (dict): additional information about the environment.
        """

        # Initialize the variables that are useful for the step
        info: dict = {}
        done: bool = self._day >= len(self._dataframe.index.unique()) - 1
        cost: float = 0

        if done:
            return self._state_memory[-1], 0, done, info

        self._raw_actions_memory.append(actions)

        previous_state = self._state_memory[-1]
        cash_amount = deepcopy(previous_state.cash_amount)
        portfolio_state = previous_state.portfolio_state.copy()
        actions_performed = actions.copy()

        # Sort the actions to do the sells first and then the buys
        argsort_actions = np.argsort(actions)

        for sorted_index in argsort_actions:
            share_price = self._dataframe.loc[self._day]["close"].values[sorted_index]
            num_shares = actions[sorted_index]

            if num_shares < 0:
                num_shares = min(abs(num_shares), portfolio_state[sorted_index])
                transaction_price = share_price * num_shares

                transaction_cost = transaction_price * self._config.fees.cost_pct
                transaction_cost = max(transaction_cost, self._config.fees.min_cost)

                cost += transaction_cost
                cash_amount += transaction_price - transaction_cost

                portfolio_state[sorted_index] -= num_shares
                actions_performed[sorted_index] = -num_shares

            elif num_shares > 0:
                transaction_price = share_price * num_shares
                transaction_cost = transaction_price * self._config.fees.cost_pct
                transaction_cost = max(transaction_cost, self._config.fees.min_cost)

                while (
                    transaction_price + transaction_cost > cash_amount
                    and num_shares > 0
                ):
                    num_shares -= 1
                    transaction_price = share_price * num_shares
                    transaction_cost = transaction_price * self._config.fees.cost_pct
                    transaction_cost = max(transaction_cost, self._config.fees.min_cost)

                cost += transaction_cost
                cash_amount -= transaction_price + transaction_cost

                portfolio_state[sorted_index] += num_shares
                actions_performed[sorted_index] = num_shares

        self._performed_actions_memory.append(actions_performed)

        # Update the state of the environment
        self._day += 1
        next_state = State(
            observable_state=self._dataframe.loc[self._day].values,
            portfolio_state=portfolio_state,
            cash_amount=cash_amount,
        )

        # Compute the total asset value and the reward as the change in the latter
        total_asset_value = cash_amount + np.sum(
            self._dataframe.loc[self._day, :]["close"] * portfolio_state
        )

        reward = total_asset_value - self._total_asset_value_memory[-1]

        self._rewards_memory.append(reward)
        self._total_asset_value_memory.append(total_asset_value)
        self._state_memory.append(next_state)

        # TODO: Investigate why there is this scaling
        reward = reward * self._config.reward_scaling

        return next_state, reward, done, info

    def reset(self) -> None:
        """Reset the environment."""
        self._episode += 1
        self._initiate_env()

    def _initiate_env(self) -> None:
        """Initialize the environment."""
        self._day = 0
        self._total_asset_value_memory = [self._config.initial_cash_amount]
        self._state_memory = [
            State(
                observable_state=self._dataframe.loc[self._day].values,
                portfolio_state=np.zeros((len(self._config.actions_ticker_list),)),
                cash_amount=self._config.initial_cash_amount,
            )
        ]
        self._rewards_memory = []
        self._performed_actions_memory = []
        self._raw_actions_memory = []
