"""Module that contains the entrypoints to the training and testing."""

import os

from testing.run_testing import main as run_testing
from training.run_training import main as run_training


def format_code():
    """Format the code using ruff."""
    os.system("bash tools/format.sh")


__all__ = ["run_training", "run_testing", "format_code"]
