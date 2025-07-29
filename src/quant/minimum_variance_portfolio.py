"""Minimum Variance Portfolio algorithm based on Modern Portfolio Theory."""

import cvxpy as cp
import torch
from torchrl.modules import SafeModule


class MinimumVariancePortfolioModule(torch.nn.Module):
    """"""

    def _compute_weights(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the weights for the minimum variance portfolio."""
        num_tickers = observation.shape[-2]
        cov = torch.cov(observation)
        return self._solve_weights(cov, num_tickers)

    @staticmethod
    def _solve_weights(cov: torch.Tensor, num_tickers: int) -> torch.Tensor:
        """Compute the weights for the minimum variance portfolio."""
        weights = cp.Variable(num_tickers)
        objective = cp.Minimize(cp.quad_form(weights, cov.detach().numpy()))
        constraints = [cp.sum(weights, axis=-1) == 1, weights >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return torch.tensor(weights.value)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """"""
        if len(observation.shape) == 3:
            weights = []
            for batch_idx in range(observation.shape[0]):
                weights.append(self._compute_weights(observation[batch_idx]))
            weights = torch.stack(weights, dim=0)

        else:
            weights = self._compute_weights(observation)
        weights = torch.round(weights, decimals=2)

        return weights


def MinimumVariancePortfolioPolicy():
    return SafeModule(
        module=MinimumVariancePortfolioModule(),
        in_keys=["observation"],
        out_keys=["action"],
    )
