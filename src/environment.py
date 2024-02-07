"""Module to define the environment to trade stocks."""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import CatTensors, Compose, DoubleToFloat, EnvBase, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import check_env_specs

from config.schemas import EnvironmentConfigSchema


class TradingEnv(EnvBase):
    """Class to define the trading environment."""

    batch_locked: bool = False

    def __init__(
        self,
        config: EnvironmentConfigSchema,
        num_tickers: int,
        env_data: pd.DataFrame,
        seed: int | None = None,
        device: str = "cpu",
    ):
        batch_size = (
            torch.Size([config.batch_size]) if config.batch_size is not None else None
        )
        super().__init__(device=device, batch_size=batch_size)
        self._env_data = env_data
        self._num_tickers = num_tickers
        self._cash_amount = config.cash_amount
        self._day = torch.zeros((), dtype=torch.int32)
        self._convert_to_list_tensors(self._env_data)

        # Should be safe to remove, keeping in the meantime for debug
        td_params = self.gen_params()
        self._make_spec(td_params)

        if seed is None:
            seed = torch.empty((), dtype=torch.int32).random_().item()
        self._set_seed(seed)

    def _convert_to_list_tensors(self, dataframe: pd.DataFrame):
        self.column_names = dataframe.columns
        self.states_per_day = []
        self._max_day = len(np.unique(dataframe.index.values)) - 1
        for index in range(self._max_day + 1):
            self.states_per_day.append(
                TensorDict(
                    {
                        column: torch.tensor(
                            data=dataframe.loc[index, column].values,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for column in self.column_names
                    },
                    batch_size=[],
                    device=self.device,
                )
            )

    def _get_state_of_day(self, day: int | torch.Tensor, batch_size: List | torch.Size):
        state = self.states_per_day[day].clone()
        if batch_size:
            for key, value in state.items():
                state[key] = value.expand(batch_size + value.shape).contiguous()
            state.batch_size = batch_size
        return state

    def _step(self, tensordict: TensorDict):
        # Check if done
        done = self._day == self._max_day

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
        reward = (
            new_cash_amount
            + torch.sum(new_num_shares_owned * out["close"], dim=-1, keepdim=True)
            - portfolio_value
        )
        out["reward"] = reward

        return out

    def _reset(self, tensordict: Optional[TensorDict] = None):
        self._day = torch.zeros((), dtype=torch.int32)

        if tensordict is None:
            tensordict = self.gen_params(batch_size=self.batch_size)
        elif tensordict.is_empty():
            tensordict = self.gen_params(batch_size=tensordict.shape)

        out = TensorDict(
            {
                "cash_amount": torch.full(
                    (*tensordict.shape, 1),
                    self._cash_amount,
                    dtype=torch.float32,
                    device=self.device,
                ),  # tensordict["initial_cash_amount"],
                "num_shares_owned": torch.zeros(
                    (*tensordict.shape, self._num_tickers),
                    dtype=torch.float32,
                    device=self.device,
                ),
            },
            batch_size=tensordict.shape,
            device=self.device,
        )

        state = self._get_state_of_day(self._day, tensordict.shape)

        out = out.update(state)
        return out

    def gen_params(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = []

        td_params = TensorDict(
            {
                "initial_cash_amount": torch.full(
                    size=(1,),
                    fill_value=self._cash_amount,
                    device=self.device,
                )
            },
            batch_size=[],
            device=self.device,
        )

        if batch_size:
            td_params = td_params.expand(batch_size).contiguous()
        return td_params

    def _set_seed(self, seed):
        self._seed = seed
        self.rng = torch.manual_seed(seed)

    def _make_spec(self, td_params):
        state = {
            key: UnboundedContinuousTensorSpec(
                shape=(self._num_tickers,),
                dtype=torch.float32,
                device=self.device,
            )
            for key in self.column_names
        }
        self.observation_spec = CompositeSpec(
            {
                "cash_amount": UnboundedContinuousTensorSpec(
                    shape=(1,),
                    device=self.device,
                ),
                "num_shares_owned": UnboundedContinuousTensorSpec(
                    shape=(self._num_tickers,),
                    device=self.device,
                ),
                **state,
            },
            shape=(),
            device=self.device,
        )
        self.state_spec = self.observation_spec.clone()

        self.action_spec = BoundedTensorSpec(
            shape=(self._num_tickers,),
            low=-1,
            high=1,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(*td_params.shape, 1),
            device=self.device,
        )


def _apply_transform_observation(env: EnvBase) -> TransformedEnv:
    """Concatenates the columns into an observation key."""
    transformed_env = TransformedEnv(
        env=env,
        transform=CatTensors(
            in_keys=["num_shares_owned", *env.column_names],
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
    num_tickers: int,
    env_data: pd.DataFrame,
    seed: int | None = None,
    device: str = "cpu",
) -> TransformedEnv:
    """Get the environment to train using SAC."""
    env = TradingEnv(config, num_tickers, env_data, seed, device)
    env = _apply_transform_observation(env)
    env = _apply_sac_transforms(env, config)
    check_env_specs(env, seed=seed)
    return env
