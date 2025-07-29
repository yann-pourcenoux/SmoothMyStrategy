"""Maximum Sharpe Ratio Portfolio algorithm based on Modern Portfolio Theory."""

import cvxpy as cp
import torch
from torchrl.modules import SafeModule


class MaxSharpeRatioPortfolioModule(torch.nn.Module):
    """Portfolio optimization module that maximizes the Sharpe ratio.

    This module finds portfolio weights that maximize the ratio of expected return
    to volatility. The risk-free rate doesn't affect the optimization since it's
    a constant when maximizing with respect to weights.
    """

    def _compute_weights(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the weights for the maximum Sharpe ratio portfolio.

        Args:
            observation: Tensor of shape (time_steps, num_tickers) containing historical returns.

        Returns:
            Portfolio weights that maximize the Sharpe ratio.
        """
        num_tickers = observation.shape[-2]

        # Calculate expected returns (mean) and covariance matrix
        expected_returns = torch.mean(observation, dim=-1)
        cov = torch.cov(observation)

        return self._solve_weights(expected_returns, cov, num_tickers)

    def _solve_weights(
        self, expected_returns: torch.Tensor, cov: torch.Tensor, num_tickers: int
    ) -> torch.Tensor:
        """Solve for the weights that maximize the Sharpe ratio.

        Args:
            expected_returns: Expected returns for each asset.
            cov: Covariance matrix of asset returns.
            num_tickers: Number of assets in the portfolio.

        Returns:
            Optimal portfolio weights.
        """
        # Convert to numpy for cvxpy
        mu = expected_returns.detach().numpy()
        sigma = cov.detach().numpy()

        # Define optimization variables
        weights = cp.Variable(num_tickers)

        # Objective: Maximize Sharpe ratio without risk-free rate
        # We maximize w^T * mu subject to w^T * Sigma * w = 1
        # This gives us the tangency portfolio on the efficient frontier
        portfolio_return = weights.T @ mu

        portfolio_variance = cp.quad_form(weights, sigma)

        # Maximize expected return subject to unit variance constraint
        objective = cp.Maximize(portfolio_return)
        constraints = [
            portfolio_variance == 1,  # Unit variance constraint (equality)
            weights >= 0,  # Long-only constraint
        ]

        problem = cp.Problem(objective, constraints)

        problem.solve()
        weights = torch.tensor(weights.value)
        weights = weights / torch.sum(weights)
        return weights

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute optimal portfolio weights.

        Args:
            observation: Historical returns tensor of shape (batch_size, time_steps, num_tickers)
                        or (time_steps, num_tickers).

        Returns:
            Portfolio weights tensor of shape (batch_size, num_tickers) or (num_tickers,).
        """
        if len(observation.shape) == 3:
            # Batch processing
            weights = []
            for batch_idx in range(observation.shape[0]):
                weights.append(self._compute_weights(observation[batch_idx]))
            weights = torch.stack(weights, dim=0)
        else:
            # Single sample processing
            weights = self._compute_weights(observation)

        # Round weights to 2 decimal places for practical implementation
        weights = torch.round(weights, decimals=2)

        return weights


def MaxSharpeRatioPortfolioPolicy():
    """Create a SafeModule for maximum Sharpe ratio portfolio optimization.

    Returns:
        SafeModule configured for portfolio optimization.
    """
    return SafeModule(
        module=MaxSharpeRatioPortfolioModule(),
        in_keys=["observation"],
        out_keys=["action"],
    )
