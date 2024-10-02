"""Module to handle the logging either to std out or to a file or to a tool."""

import os
from typing import Any, Optional

import pandas as pd
import torch
import wandb
from torchrl.record.loggers.wandb import WandbLogger

from common.config import ExperimentConfigSchema


def log_metrics(logger, metrics, step: Optional[int] = None):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def log_figure(figure: Any, title: str):
    wandb.log({title: wandb.Image(figure)})


def log_model(model: torch.nn.Module) -> None:
    policy_path = os.path.join(wandb.run.dir, "model.pth")
    torch.save(model, policy_path)
    wandb.save(policy_path)


def get_logger(config: ExperimentConfigSchema) -> WandbLogger:
    """Get a wandb logger."""
    logger = WandbLogger(
        exp_name=config.logging.experiment,
        offline=not config.logging.online,
        save_dir=config.logging.logging_directory,
        project=config.logging.project,
        config=config,
    )
    return logger


def log_df(df: pd.DataFrame, title: str):
    """Log a pandas DataFrame to wandb."""
    df.to_csv(os.path.join(wandb.run.dir, f"{title}.csv"))
    wandb.log({title: wandb.Table(dataframe=df)})
