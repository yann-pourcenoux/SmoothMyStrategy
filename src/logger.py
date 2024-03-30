"""Module to handle the logging either to std out or to a file or to a tool."""

from typing import Any

import wandb
from torchrl.record.loggers.wandb import WandbLogger

from config.schemas import ExperimentConfigSchema


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def log_figure(figure: Any, title: str):
    wandb.log({title: wandb.Image(figure)})


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
