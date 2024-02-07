"""Module that holds all the utils function to run the training."""

from typing import Any

import torch
import wandb
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor
from torchrl.record.loggers.wandb import WandbLogger

from config.schemas import (
    CollectorConfigSchema,
    LoggingConfigSchema,
    ReplayBufferConfigSchema,
)


def make_collector(
    train_env: EnvBase,
    actor_model_explore: ProbabilisticActor,
    config: CollectorConfigSchema,
    device: str = "cpu",
    seed: int = 0,
) -> SyncDataCollector:
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=config.init_random_frames,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        storing_device="cpu",
        env_device=device,
        policy_device=device,
    )
    collector.set_seed(seed)
    return collector


def make_replay_buffer(
    config: ReplayBufferConfigSchema,
) -> TensorDictPrioritizedReplayBuffer | TensorDictReplayBuffer:
    """Make replay buffer."""
    storage = LazyMemmapStorage(
        config.buffer_size,
        scratch_dir=config.buffer_scratch_dir,
        device="cpu",
    )
    if config.prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=config.alpha,
            beta=config.beta,
            pin_memory=config.pin_memory,
            prefetch=config.prefetch,
            storage=storage,
            batch_size=config.batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=config.pin_memory,
            prefetch=config.prefetch,
            storage=storage,
            batch_size=config.batch_size,
        )
    return replay_buffer


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def log_figure(figure: Any, title: str):
    wandb.log({title: wandb.Image(figure)})


def get_logger(config: LoggingConfigSchema) -> WandbLogger:
    """Get a wandb logger."""
    logger = WandbLogger(
        exp_name=config.experiment,
        offline=not config.online,
        save_dir=config.logging_directory,
        project=config.project,
    )
    return logger


def compute_portfolio_value(tensordict: TensorDict) -> list[float] | list[list[float]]:
    """Compute the value of a portfolio over the time steps and batch dimensions."""
    portfolio_value = tensordict["cash_amount"] + torch.sum(
        tensordict["num_shares_owned"] * tensordict["close"],
        dim=-1,
        keepdim=False,
    )
    if len(portfolio_value.shape) == 2:
        portfolio_value = portfolio_value.mean(0)
    return portfolio_value.tolist()
