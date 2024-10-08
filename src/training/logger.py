"""Module to handle the logging either to std out or to a file or to a tool."""

import os
from typing import Any

import pandas as pd
import torch
import wandb


def log_figure(figure: Any, title: str):
    wandb.log({title: wandb.Image(figure)})


def log_model(model: torch.nn.Module) -> None:
    """Save and log the model to W&B in the run directory."""
    # Ensure that a W&B run is active
    if wandb.run is None:
        raise ValueError(
            "W&B run is not initialized. Call wandb.init() before logging models."
        )

    # Define the path within the W&B run directory
    policy_path = os.path.join(wandb.run.dir, "model.pth")

    # Save the model
    torch.save(model, policy_path)

    # Log the model file to W&B
    wandb.save(policy_path)


def log_df(df: pd.DataFrame, title: str) -> None:
    """Save and log a pandas DataFrame to W&B in the run directory."""
    # Ensure that a W&B run is active
    if wandb.run is None:
        raise ValueError(
            "W&B run is not initialized. Call wandb.init() before logging dataframes."
        )

    # Define the CSV path within the W&B run directory
    csv_path = os.path.join(wandb.run.dir, f"{title}.csv")

    # Save the DataFrame to CSV
    df.to_csv(csv_path)

    # Log the DataFrame as a Table to W&B
    wandb.log({title: wandb.Table(dataframe=df)})
