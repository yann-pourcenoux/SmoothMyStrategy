"""Algorithm that buys one share everyday."""

import torch
from tensordict import TensorDict

from quant.base import TraditionalAlgorithm


class BuySharesModule(TraditionalAlgorithm):
    """Module that implements a simple buy-shares strategy."""

    def __init__(self):
        """Initialize the BuySharesModule."""
        super().__init__()
        self.a = 0

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        """Implement a strategy to buy shares for all tickers.

        Args:
            tensordict: Tensor containing closing prices.

        Returns:
            Tensor containing action values (buy one share for each ticker).
        """
        # Action is +1 share for all tickers
        if not self.a:
            print(tensordict)
            self.a = 1
        return torch.ones_like(tensordict) * 42
