"""Base class for traditional trading algorithms."""

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import SafeModule


class TraditionalAlgorithm(ABC, torch.nn.Module):
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

    def __init__(self, algorithm: nn.Module):
        # SafeModule requires in_keys and out_keys, but we don't strictly need them
        # as the logic is custom in _forward. Pass dummy ones.
        # Using "observation" as in_key based on typical RL policy usage.
        super().__init__(module=algorithm, in_keys=["observation"], out_keys=["action"])
        self.algorithm = algorithm

    def _forward(self, tensordict: TensorDict) -> TensorDict:
        """Process the input tensordict and produce an action tensordict.

        Args:
            tensordict: The input TensorDict containing observation data.

        Returns:
            TensorDict: The output TensorDict with an action key.
        """
        return
        action = self.algorithm(tensordict["something that does not exist"])
        tensordict = tensordict.set("action", action)
        return tensordict
