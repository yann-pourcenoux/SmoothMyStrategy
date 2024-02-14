"""Module to define the environment to trade stocks."""

from typing import List

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import CatTensors, Compose, DoubleToFloat, EnvBase, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import check_env_specs

from config.schemas import EnvironmentConfigSchema
from data.container import DataContainer


class TradingEnv(EnvBase):
    """Class to define the trading environment."""

    batch_locked: bool = False

    def __init__(
        self,
        config: EnvironmentConfigSchema,
        data_container: DataContainer,
        fixed_initial_distribution: bool = False,
        seed: int | None = None,
        device: str = "cpu",
    ):
        batch_size = [config.batch_size] if config.batch_size is not None else None
        super().__init__(device=device, batch_size=batch_size)
        self._data_container = data_container
        self._env_data = data_container.data
        self._fixed_initial_distribution = fixed_initial_distribution
        self._num_tickers = data_container.num_tickers
        self._num_time_steps = data_container.num_time_steps
        self._cash_amount = config.cash_amount
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
        for index in range(self._num_time_steps):
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

    def _get_state_of_day(self, day: int | torch.Tensor, batch_size: List | torch.Size):
        state = self.states_per_day[day].clone()
        if batch_size:
            for key, value in state.items():
                state[key] = value.expand(batch_size + value.shape).contiguous()
            state.batch_size = batch_size
        return state

    def _step(self, tensordict: TensorDict):
        # Check if done
        done = self._day == self._num_time_steps - 1

        if done:
            return TensorDict(
                {
                    **self._get_state_of_day(self._day, tensordict.shape),
                    "cash_amount": tensordict["cash_amount"],
                    "num_shares_owned": tensordict["num_shares_owned"],
                    "reward": torch.zeros_like(tensordict["cash_amount"]),
                    "done": torch.ones_like(
                        tensordict["cash_amount"], dtype=torch.bool
                    ),
                },
                batch_size=tensordict.shape,
            )

        actions = tensordict["action"] * 100

        # Compute portfolio value
        portfolio_value = tensordict["cash_amount"] + torch.sum(
            tensordict["num_shares_owned"] * tensordict["close"], dim=-1, keepdim=True
        )

        new_num_shares_owned = tensordict["num_shares_owned"].clone()
        new_cash_amount = tensordict["cash_amount"].clone()

        sorted_indices = torch.argsort(actions, dim=-1, descending=False)
        for indices in torch.t(sorted_indices):
            indices = torch.unsqueeze(indices, dim=-1)
            num_shares_owned = torch.gather(
                tensordict["num_shares_owned"], dim=-1, index=indices
            )
            price_share = torch.gather(tensordict["close"], dim=-1, index=indices)

            with torch.no_grad():
                min_num_shares_action = -num_shares_owned
                max_num_shares_action = new_cash_amount // price_share

            num_shares_action = torch.clamp(
                input=torch.gather(actions, dim=-1, index=indices),
                min=min_num_shares_action,
                max=max_num_shares_action,
            )

            num_shares_owned += num_shares_action
            new_cash_amount -= price_share * num_shares_action
            new_num_shares_owned.scatter_(dim=-1, index=indices, src=num_shares_owned)

        self._day += 1

        out = TensorDict(
            {
                **self._get_state_of_day(self._day, tensordict.shape),
                "cash_amount": new_cash_amount,
                "num_shares_owned": new_num_shares_owned,
                "done": torch.zeros_like(tensordict["cash_amount"], dtype=torch.bool),
            },
            batch_size=tensordict.shape,
        )

        # Compute reward
        new_portfolio_value = new_cash_amount + torch.sum(
            new_num_shares_owned * out["close"], dim=-1, keepdim=True
        )
        reward = torch.log(new_portfolio_value / portfolio_value)
        out["reward"] = reward

        return out

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        del tensordict
        self._day = torch.zeros((), dtype=torch.int32)
        out = self._get_state_of_day(self._day, self.batch_size)

        if self._fixed_initial_distribution:
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
        num_shares_owned = torch.floor(distribution * self._cash_amount / out["close"])
        # Update the cash amount
        cash_amount = self._cash_amount - torch.sum(
            num_shares_owned * out["close"], dim=-1, keepdim=True
        )

        out["cash_amount"] = cash_amount
        out["num_shares_owned"] = num_shares_owned
        return out

    def _set_seed(self, seed):
        self._seed = seed
        self.rng = torch.manual_seed(seed)

    def _make_spec(self):  # , td_params):
        state = {
            key: UnboundedContinuousTensorSpec(
                shape=self.batch_size + (self._num_tickers,),
                dtype=torch.float32,
                device=self.device,
            )
            for key in self.column_names
        }
        self.observation_spec = CompositeSpec(
            {
                "cash_amount": UnboundedContinuousTensorSpec(
                    shape=self.batch_size + (1,),
                    device=self.device,
                ),
                "num_shares_owned": UnboundedContinuousTensorSpec(
                    shape=self.batch_size + (self._num_tickers,),
                    device=self.device,
                ),
                **state,
            },
            shape=self.batch_size,
            device=self.device,
        )
        self.state_spec = self.observation_spec.clone()

        self.action_spec = BoundedTensorSpec(
            shape=self.batch_size + (self._num_tickers,),
            low=-1,
            high=1,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=self.batch_size + (1,),
            device=self.device,
        )


def _apply_transform_observation(env: TradingEnv) -> TransformedEnv:
    """Concatenates the columns into an observation key."""
    transformed_env = TransformedEnv(
        env=env,
        transform=CatTensors(
            in_keys=env._data_container._preprocessing_config.technical_indicators,
            dim=-1,
            out_key="observation",
            del_keys=False,
        ),
        device=env.device,
    )
    return transformed_env


def _apply_sac_transforms(
    env: EnvBase, config: EnvironmentConfigSchema
) -> TransformedEnv:
    """Apply the necessary transforms to train using SAC."""
    transformed_env = TransformedEnv(
        env=env,
        transform=Compose(
            InitTracker(),
            StepCounter(config.max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
        device=env.device,
    )
    return transformed_env


def get_sac_environment(
    config: EnvironmentConfigSchema,
    data_container: DataContainer,
    fixed_initial_distribution: bool = False,
    seed: int | None = None,
    device: str = "cpu",
) -> TransformedEnv:
    """Get the environment to train using SAC."""
    env = TradingEnv(config, data_container, fixed_initial_distribution, seed, device)
    check_env_specs(env, seed=seed)
    env = _apply_transform_observation(env)
    check_env_specs(env, seed=seed)
    env = _apply_sac_transforms(env, config)
    check_env_specs(env, seed=seed)
    return env
