"""Module that contains the entrypoints to the training and testing."""

import os

from evaluation.run_evaluation import main as run_evaluation
from rl.run_training import main as run_training


def format_code():
    """Format the code using ruff."""
    os.system("bash tools/format.sh")


__all__ = ["format_code", "run_evaluation", "run_training"]
