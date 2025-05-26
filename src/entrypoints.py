"""Module that contains the entrypoints to the training and testing."""

import os

from evaluation.run_evaluation import main as run_evaluation
from rl.run_training import main as run_training


def format():
    """Format the code using ruff."""
    os.system("bash tools/format.sh")


def run_test():
    """Run the tests."""
    os.system("pytest -n auto src")


def run_visualization():
    """Format the code using ruff."""
    os.system("streamlit run src/visualization/visualize.py")


__all__ = ["format", "run_evaluation", "run_training", "run_visualization", "run_test"]
