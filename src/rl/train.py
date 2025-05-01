"""Module to train an agent."""

import time
from typing import Dict

import tqdm
from tensordict import TensorDict
from torch import optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer
from torchrl.objectives import LossModule, SoftUpdate

from config import TrainingConfigSchema


def train(
    config: TrainingConfigSchema,
    replay_buffer: ReplayBuffer | PrioritizedReplayBuffer,
    collector: SyncDataCollector,
    loss_module: LossModule,
    optimizers: tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer],
    target_net_updater: SoftUpdate,
    schedulers: list,
) -> Dict[str, float]:
    optimizer_actor, optimizer_critic, optimizer_alpha = optimizers
    metrics_to_log = {}
    # Update the models
    training_start = time.time()
    losses_values = TensorDict({}, batch_size=[config.num_steps_per_epoch])
    for i in tqdm.tqdm(
        range(config.num_steps_per_epoch),
        desc="Optimization",
        unit="step",
        leave=False,
    ):
        # Sample from replay buffer
        sampled_tensordict = replay_buffer.sample()
        if sampled_tensordict.device != collector.env_device:
            sampled_tensordict = sampled_tensordict.to(
                collector.env_device, non_blocking=True
            )
        else:
            sampled_tensordict = sampled_tensordict.clone()
        # Compute loss
        loss_td = loss_module(sampled_tensordict)
        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]
        # Update actor
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()
        # Update critic
        optimizer_critic.zero_grad()
        q_loss.backward()
        optimizer_critic.step()
        # Update alpha
        optimizer_alpha.zero_grad()
        alpha_loss.backward()
        optimizer_alpha.step()

        for scheduler in schedulers:
            scheduler.step()

        losses_values[i] = loss_td.select(
            "loss_actor", "loss_qvalue", "loss_alpha"
        ).detach()
        # Update qnet_target params
        target_net_updater.step()
        # Update priority
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            replay_buffer.update_priority(sampled_tensordict)
    training_time = time.time() - training_start

    metrics_to_log["train/q_loss"] = losses_values.get("loss_qvalue").mean().item()
    metrics_to_log["train/actor_loss"] = losses_values.get("loss_actor").mean().item()
    metrics_to_log["train/alpha_loss"] = losses_values.get("loss_alpha").mean().item()
    metrics_to_log["train/alpha"] = loss_td["alpha"].item()
    metrics_to_log["train/entropy"] = loss_td["entropy"].item()
    metrics_to_log["timer/train/training_time"] = training_time
    return metrics_to_log
