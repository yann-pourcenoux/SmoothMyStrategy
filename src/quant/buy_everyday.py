"""Algorithm that buys one share everyday."""

import torch
from torchrl.modules import SafeModule


class BuySharesModule(torch.nn.Module):
    """Module that implements a simple buy-shares strategy."""

    def __init__(self):
        """Initialize the BuySharesModule."""
        super().__init__()

    def forward(self, num_shares_owned: torch.Tensor) -> torch.Tensor:
        """Implement a strategy to buy shares for all tickers.

        Args:
            num_shares_owned (torch.Tensor): Tensor containing number of shares owned.

        Returns:
            Tensor containing action values (buy one share for each ticker).
        """
        return torch.ones_like(num_shares_owned)


def BuySharesPolicy():
    return SafeModule(module=BuySharesModule(), in_keys=["num_shares_owned"], out_keys=["action"])
