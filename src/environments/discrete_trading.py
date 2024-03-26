"""Module that contains the class to define the trading environment."""

import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec

from environments.trading import TradingEnv


class DiscreteTradingEnv(TradingEnv):
    """Class to define the trading environment when actions are discrete.

    The actions are a distribution over the tickers, where the agent can choose to buy,
    sell or hold a certain amount of shares of a certain ticker.
    """

    def _process_actions(self, tensordict: TensorDict):
        """Mapping the actions."""
        actions = tensordict["action"]
        # Values in [0, 1 ,2]
        actions = torch.argmax(actions, dim=-1)
        # Rescale to [-1, 0, 1]
        actions = actions - 1
        return tensordict["action"] * 100

    def _get_action_spec(self):
        return BoundedTensorSpec(
            shape=self.batch_size + (self._num_tickers, 3),
            low=0,
            high=1,
            device=self.device,
        )
