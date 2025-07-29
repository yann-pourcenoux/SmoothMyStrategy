"""Module to define the environment to trade stocks."""

from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase

from config.environment import EnvironmentConfigSchema
from data.container import DataContainer


class BaseTradingEnv(EnvBase):
    """Class to define the base trading environment.

    The actions are supposed to a number of shares to buy or sell.
    It is on the child class to define the action space and define the actions mapping.
    """

    batch_locked: bool = False

    def __init__(
        self,
        config: EnvironmentConfigSchema,
        data_container: DataContainer,
        seed: int | None = None,
        device: str = "cpu",
    ):
        batch_size = torch.Size([config.batch_size]) if config.batch_size is not None else None
        super().__init__(device=device, batch_size=batch_size)

        self._config = config
        self._env_data, self._num_time_steps, self._dates, self._tickers = (
            data_container.get_env_data(config.start_date, config.end_date)
        )
        self._num_tickers = data_container.num_tickers
        self.technical_indicators = data_container._preprocessing_config.technical_indicators

        # Set up
        self._day = torch.zeros((), dtype=torch.int32)
        self._convert_to_list_tensors(self._env_data)

        # Should be safe to remove, keeping in the meantime for debug
        self._make_spec()

        if seed is None:
            seed = torch.empty((), dtype=torch.int32).random_().item()
        self._set_seed(seed)

    @abstractmethod
    def _convert_actions(self, tensordict: TensorDict):
        pass

    @abstractmethod
    def _get_action_spec(self):
        pass

    # TODO(@yann.pourcenoux): Maybe this could be improved
    def _convert_to_list_tensors(self, dataframe: pd.DataFrame):
        self.column_names = dataframe.columns
        self.states_per_day = []
        for index in range(self._num_time_steps + 1):
            tensordict = TensorDict(
                {},
                batch_size=[],
                device=self.device,
            )
            for column in self.column_names:
                data = dataframe.loc[index, column]
                if self._num_tickers > 1:
                    data = data.values
                else:
                    data = np.array([data])

                if data.dtype in [np.float32, np.float64]:
                    data = data.astype(np.float32)
                elif data.dtype in [np.int32, np.int64]:
                    data = data.astype(np.int32)

                tensordict[column] = torch.tensor(data=data, device=self.device)

            self.states_per_day.append(tensordict)

    def _get_state_of_day(self, day: int | torch.Tensor, batch_size: list[int] | torch.Size):
        state = self.states_per_day[day].clone()
        if batch_size:
            for key, value in state.items():
                state[key] = value.expand(batch_size + value.shape).contiguous()
            state.batch_size = batch_size
        return state

    def _step(self, tensordict: TensorDict):
        """Perform the step of the environment."""
        return self._perform_trading_action(tensordict)

    def _perform_trading_action(self, tensordict: TensorDict):
        """Perform the trading action."""
        # Check if done, it will be used at the bottom
        done = self._day == self._num_time_steps - 1

        condition = tensordict["is_first_day_of_month"][..., 0:1]  # Keep the last dimension
        tensordict["deposit"] = torch.where(condition, self._config.monthly_cash, 0)

        tensordict["cash"] = torch.where(
            condition,
            tensordict["cash"] + self._config.monthly_cash,
            tensordict["cash"],
        )

        # Compute portfolio value
        portfolio_value = tensordict["cash"] + torch.sum(
            tensordict["num_shares_owned"] * tensordict["adj_close"],
            dim=-1,
            keepdim=True,
        )

        new_num_shares_owned = tensordict["num_shares_owned"].clone()
        new_cash = tensordict["cash"].clone()

        actions = self._convert_actions(tensordict)["action"]

        sorted_indices = torch.argsort(actions, dim=-1, descending=False)
        for indices in torch.t(sorted_indices):
            indices = torch.unsqueeze(indices, dim=-1)
            num_shares_owned = torch.gather(tensordict["num_shares_owned"], dim=-1, index=indices)
            price_share = torch.gather(tensordict["adj_close"], dim=-1, index=indices)

            with torch.no_grad():
                min_num_shares_action = -num_shares_owned
                max_num_shares_action = new_cash / price_share

            num_shares_action = torch.clamp(
                input=torch.gather(actions, dim=-1, index=indices),
                min=min_num_shares_action,
                max=max_num_shares_action,
            )

            num_shares_owned += num_shares_action
            new_cash -= price_share * num_shares_action
            new_num_shares_owned.scatter_(dim=-1, index=indices, src=num_shares_owned)

        self._day += 1

        out = TensorDict(
            {
                **self._get_state_of_day(self._day, tensordict.shape),
                "cash": new_cash,
                "num_shares_owned": new_num_shares_owned,
                "deposit": tensordict["deposit"],
            },
            batch_size=tensordict.shape,
            device=self.device,
        )

        # Compute reward
        new_portfolio_value = new_cash + torch.sum(
            new_num_shares_owned * out["adj_close"], dim=-1, keepdim=True
        )
        reward = torch.log(new_portfolio_value / portfolio_value)

        # Compute new reward
        returns_assets = torch.mean(
            out["adj_close"] / tensordict["adj_close"], dim=-1, keepdim=True
        )
        reward = reward - torch.log(returns_assets)

        out["reward"] = reward

        # Set the done and terminated
        if done:
            out["done"] = torch.ones_like(tensordict["cash"], dtype=torch.bool)
        else:
            out["done"] = torch.zeros_like(tensordict["cash"], dtype=torch.bool)

        return out

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        del tensordict  # Not used in that function
        self._day = torch.zeros((), dtype=torch.int32)
        out = self._get_state_of_day(self._day, self.batch_size)

        # +1 comes form the cash
        distribution_args = {
            "size": self.batch_size + (self._num_tickers + 1,),
            "dtype": torch.float32,
            "device": self.device,
        }
        if self._config.random_initial_distribution is not None:
            distribution = (
                torch.rand(**distribution_args) * self._config.random_initial_distribution
            )
            distribution = torch.softmax(distribution, dim=-1)
        else:
            distribution = torch.ones(**distribution_args)
            distribution = distribution / torch.sum(distribution, dim=-1, keepdim=True)

        # Take only the distribution of shares
        distribution = distribution[..., :-1]
        # Compute the number of shares
        num_shares_owned = torch.floor(distribution * self._config.cash / out["adj_close"])
        # Update the cash amount
        cash = self._config.cash - torch.sum(
            num_shares_owned * out["adj_close"], dim=-1, keepdim=True
        )

        out["cash"] = cash
        out["deposit"] = torch.ones_like(cash) * self._config.cash
        out["num_shares_owned"] = num_shares_owned
        return out

    def _set_seed(self, seed):
        self._seed = seed
        self.rng = torch.manual_seed(seed)

    def _make_spec(self):
        # Determine dtypes from the first day's data
        sample_state = self.states_per_day[0]
        state = {
            key: Unbounded(
                shape=self.batch_size + (self._num_tickers,),
                dtype=sample_state[key].dtype,
                device=self.device,
            )
            for key in self.column_names
        }
        self.observation_spec = Composite(
            {
                "cash": Unbounded(
                    shape=self.batch_size + (1,),
                    device=self.device,
                ),
                "deposit": Unbounded(
                    shape=self.batch_size + (1,),
                    device=self.device,
                ),
                "num_shares_owned": Unbounded(
                    shape=self.batch_size + (self._num_tickers,),
                    device=self.device,
                ),
                **state,
            },
            shape=self.batch_size,
            device=self.device,
        )
        self.state_spec = self.observation_spec.clone()

        self.action_spec = self._get_action_spec()

        self.reward_spec = Unbounded(
            shape=self.batch_size + (1,),
            device=self.device,
        )

    def process_rollout(self, rollout: TensorDict) -> pd.DataFrame:
        """Perform a rollout and return a DataFrame with the results."""
        num_steps = rollout.shape[-1]

        dates = self._dates[:num_steps]
        tickers = self._tickers

        # Take only the value for the first batch
        close = rollout["adj_close"].cpu().numpy()[0]
        num_shares_owned = rollout["num_shares_owned"].cpu().numpy()[0]
        actions = rollout["action"].cpu().numpy()[0]

        cash = rollout["cash"].cpu().numpy()[0, :, 0]
        deposit = rollout["deposit"].cpu().numpy()[0, :, 0]

        data = {
            "date": dates,
            "cash": cash,
            "deposit": deposit,
        }
        for i, ticker in enumerate(tickers):
            data[f"action_{ticker}"] = actions[:, i]
            data[f"close_{ticker}"] = close[:, i]
            data[f"num_shares_owned_{ticker}"] = num_shares_owned[:, i]

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
