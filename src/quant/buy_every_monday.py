"""Algorithm that buys one share everyday."""

import torch


class BuySharesModule(torch.nn.Module):
    """Module that implements a simple buy-shares strategy."""

    def __init__(self, scaling_factor: float = 1.0):
        """Initialize the BuySharesModule.

        Args:
            scaling_factor: Factor to scale the raw action values by (default: 1.0).
        """
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, adj_close: torch.Tensor) -> torch.Tensor:
        """Implement a strategy to buy shares for all tickers.

        Args:
            adj_close: Tensor containing closing prices.

        Returns:
            Tensor containing action values (buy one share for each ticker).
        """
        # Action is +1 share for all tickers (scaled by the scaling factor)
        return torch.ones_like(adj_close) / 100
