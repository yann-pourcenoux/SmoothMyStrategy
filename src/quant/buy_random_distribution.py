"""Algorithm that buys a random distribution of shares."""

import torch
from torchrl.modules import SafeModule


class BuyRandomDistributionModule(torch.nn.Module):
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
        raw_distribution = torch.rand_like(num_shares_owned)
        distribution = raw_distribution / torch.sum(raw_distribution, dim=-1, keepdim=True)
        return distribution


def BuyRandomDistributionPolicy():
    return SafeModule(
        module=BuyRandomDistributionModule(), in_keys=["num_shares_owned"], out_keys=["action"]
    )
