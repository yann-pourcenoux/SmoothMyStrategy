"""Base class for traditional trading algorithms."""

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import SafeModule


class TraditionalAlgorithm(ABC):
    """Base class for traditional trading algorithms."""

    @abstractmethod
    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        """Calculate trading actions based on the strategy.

        Args:
            tensordict: The current state represented as a TensorDict.

        Returns:
            torch.Tensor: A tensor representing the raw action to take.
        """
        pass


class TraditionalAlgorithmPolicyWrapper(SafeModule):
    """Wraps a TraditionalAlgorithm to make it compatible with the evaluation loop."""

    def __init__(self, algorithm: nn.Module, action_scaling: float = 1.0):
        # SafeModule requires in_keys and out_keys, but we don't strictly need them
        # as the logic is custom in _forward. Pass dummy ones.
        # Using "observation" as in_key based on typical RL policy usage.
        super().__init__(module=algorithm, in_keys=["adj_close"], out_keys=["action"])
        self.algorithm = algorithm
        # Scaling factor used in TradingEnv._process_actions
        self.action_scaling = action_scaling

    def _forward(self, tensordict: TensorDict) -> TensorDict:
        """Process the input tensordict and produce an action tensordict.

        Args:
            tensordict: The input TensorDict containing observation data.

        Returns:
            TensorDict: The output TensorDict with an action key.
        """
        # For PyTorch modules with a straightforward forward signature like BuySharesModule
        if hasattr(self.algorithm, "forward") and callable(self.algorithm.forward):
            if "adj_close" in tensordict:
                adj_close = tensordict["adj_close"]
                action = self.algorithm(adj_close)
                tensordict = tensordict.set("action", action / self.action_scaling)
            else:
                # Fallback for cases where adj_close might not be directly available
                # but is nested in the observation
                action = torch.ones_like(tensordict["observation"][:, :1])
                tensordict = tensordict.set("action", action / self.action_scaling)

        return tensordict
