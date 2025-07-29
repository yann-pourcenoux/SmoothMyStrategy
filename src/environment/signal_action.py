"""Module to define the environment to trade stocks."""

from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded

from environment.base import BaseTradingEnv


class SignalActionEnv(BaseTradingEnv):
    """Class to define the trading environment."""

    batch_locked: bool = False

    def _convert_actions(self, tensordict: TensorDict) -> Tensor:
        return tensordict

    def _get_action_spec(self) -> Bounded:
        return Bounded(
            shape=self.batch_size + (self._num_tickers,),
            low=-100,
            high=100,
            device=self.device,
        )
