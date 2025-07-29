"""Module to define the environment with weight-based actions."""

from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded

from environment.base import BaseTradingEnv


class WeightActionEnv(BaseTradingEnv):
    """Class to define the weight-based action environment.

    Input actions represent portfolio weights (distribution) rather than direct trading
    signals. The actions are normalized weights that determine how to allocate available
    cash across assets.
    """

    batch_locked: bool = False

    def _convert_actions(self, tensordict: TensorDict) -> Tensor:
        """Convert portfolio weights to number of shares to buy.

        Args:
            tensordict: TensorDict containing action weights and market data.

        Returns:
            TensorDict with processed actions representing shares to purchase.
        """
        # Convert weights to shares: weight * available_cash / price_per_share
        tensordict["action"] = tensordict["action"] * tensordict["cash"] / tensordict["adj_close"]
        return tensordict

    def _get_action_spec(self) -> Bounded:
        """Define the action specification for portfolio weights.

        Returns:
            Bounded action space with weights between 0 and 1 for each ticker.
        """
        return Bounded(
            shape=self.batch_size + (self._num_tickers,),
            low=0,
            high=1,
            device=self.device,
        )
