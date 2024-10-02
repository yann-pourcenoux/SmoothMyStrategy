"""Module that contains the entrypoints to the training and testing."""

from testing.run_testing import main as run_testing
from training.run_training import main as run_training

__all__ = ["run_training", "run_testing"]
