"""Module to define the environment to trade stocks."""

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import CatTensors, Compose, DoubleToFloat, EnvBase, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, VecNorm

from data.container import DataContainer
from environments.config import EnvironmentConfigSchema


class TradingEnv(EnvBase):
    """Class to define the trading environment."""

    batch_locked: bool = False

    def __init__(
        self,
        config: EnvironmentConfigSchema,
        data_container: DataContainer,
        seed: int | None = None,
        device: str = "cpu",
    ):
        batch_size = (
            torch.Size([config.batch_size]) if config.batch_size is not None else None
        )
        super().__init__(device=device, batch_size=batch_size)

        self._config = config
        self._env_data, self._num_time_steps, self._dates, self._tickers = (
            data_container.get_env_data(config.start_date, config.end_date)
        )
        self._num_tickers = data_container.num_tickers
        self.technical_indicators = (
            data_container._preprocessing_config.technical_indicators
        )

        # Set up
        self._day = torch.zeros((), dtype=torch.int32)
        self._convert_to_list_tensors(self._env_data)

        # Should be safe to remove, keeping in the meantime for debug
        self._make_spec()

        if seed is None:
            seed = torch.empty((), dtype=torch.int32).random_().item()
        self._set_seed(seed)

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
                tensordict[column] = torch.tensor(
                    data=data,
                    dtype=torch.float32,
                    device=self.device,
                )
            self.states_per_day.append(tensordict)

    def _get_state_of_day(
        self, day: int | torch.Tensor, batch_size: list[int] | torch.Size
    ):
        state = self.states_per_day[day].clone()
        if batch_size:
            for key, value in state.items():
                state[key] = value.expand(batch_size + value.shape).contiguous()
            state.batch_size = batch_size
        return state

    def _step(self, tensordict: TensorDict):
        """Perform the step of the environment."""
        tensordict["action"] = self._process_actions(tensordict)
        return self._perform_trading_action(tensordict)

    def _process_actions(self, tensordict: TensorDict):
        return tensordict["action"] * 100

    def _get_action_spec(self):
        return Bounded(
            shape=self.batch_size + (self._num_tickers,),
            low=-1,
            high=1,
            device=self.device,
        )

    def _perform_trading_action(self, tensordict: TensorDict):
        """Perform the trading action."""
        # Check if done, it will be used at the bottom
        done = self._day == self._num_time_steps - 1

        actions = tensordict["action"]

        # Compute portfolio value
        portfolio_value = tensordict["cash"] + torch.sum(
            tensordict["num_shares_owned"] * tensordict["adj_close"],
            dim=-1,
            keepdim=True,
        )

        new_num_shares_owned = tensordict["num_shares_owned"].clone()
        new_cash = tensordict["cash"].clone()

        sorted_indices = torch.argsort(actions, dim=-1, descending=False)
        for indices in torch.t(sorted_indices):
            indices = torch.unsqueeze(indices, dim=-1)
            num_shares_owned = torch.gather(
                tensordict["num_shares_owned"], dim=-1, index=indices
            )
            price_share = torch.gather(tensordict["adj_close"], dim=-1, index=indices)

            with torch.no_grad():
                min_num_shares_action = -num_shares_owned
                max_num_shares_action = new_cash // price_share

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

        if self._config.fixed_initial_distribution:
            distribution_gen_fn = torch.ones
        else:
            distribution_gen_fn = torch.rand

        # +1 comes form the free cash to keep
        distribution = distribution_gen_fn(
            size=self.batch_size + (self._num_tickers + 1,),
            dtype=torch.float32,
            device=self.device,
        )
        distribution = distribution / torch.sum(distribution, dim=-1, keepdim=True)
        # Take only the distribution of shares
        distribution = distribution[..., :-1]
        # Compute the number of shares
        num_shares_owned = torch.floor(
            distribution * self._config.cash / out["adj_close"]
        )
        # Update the cash amount
        cash = self._config.cash - torch.sum(
            num_shares_owned * out["adj_close"], dim=-1, keepdim=True
        )

        out["cash"] = cash
        out["num_shares_owned"] = num_shares_owned
        return out

    def _set_seed(self, seed):
        self._seed = seed
        self.rng = torch.manual_seed(seed)

    def _make_spec(self):
        state = {
            key: Unbounded(
                shape=self.batch_size + (self._num_tickers,),
                dtype=torch.float32,
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

        close = rollout["adj_close"]
        num_shares_owned = rollout["num_shares_owned"]
        cash = rollout["cash"]
        actions = rollout["action"]
        portfolio_value = torch.squeeze(
            cash + torch.sum(close * num_shares_owned, dim=-1, keepdim=True),
            axis=-1,
        )

        # Convert the tensors to cpu().numpy()
        close = close.cpu().numpy()[0]
        num_shares_owned = num_shares_owned.cpu().numpy()[0]
        cash = cash.cpu().numpy()[0, :, 0]
        portfolio_value = portfolio_value.cpu().numpy()[0]
        actions = actions.cpu().numpy()[0]

        data = {"date": dates, "cash": cash, "portfolio_value": portfolio_value}
        for i, ticker in enumerate(tickers):
            data[f"action_{ticker}"] = actions[:, i]
            data[f"close_{ticker}"] = close[:, i]
            data[f"num_shares_owned_{ticker}"] = num_shares_owned[:, i]

        df = pd.DataFrame(data)
        df["daily_returns"] = df["portfolio_value"].pct_change().fillna(0)
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)
        return df


def _apply_transform_observation(env: TradingEnv) -> TransformedEnv:
    """Concatenates the columns into an observation key."""
    transformed_env = TransformedEnv(
        env=env,
        transform=Compose(
            CatTensors(
                in_keys=env.technical_indicators,
                dim=-1,
                out_key="observation",
                del_keys=False,
            ),
            VecNorm(in_keys=["observation"]),
        ),
        device=env.device,
    )
    return transformed_env


def _apply_transforms(env: EnvBase) -> TransformedEnv:
    """Apply the necessary transforms to train using SAC."""
    transformed_env = TransformedEnv(
        env=env,
        transform=Compose(
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),
            RewardSum(),
        ),
        device=env.device,
    )
    return transformed_env


def apply_transforms(env: EnvBase) -> TransformedEnv:
    """Get the environment to train using SAC."""
    env = _apply_transform_observation(env)
    env = _apply_transforms(env)
    return env
