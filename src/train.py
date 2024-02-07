"""Module to train an agent."""

import time

import torch
import tqdm
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torch import optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import LossModule, SoftUpdate
from torchrl.record.loggers.wandb import WandbLogger

import utils
from config.schemas import ExperimentConfigSchema


def train_and_eval(
    collector: SyncDataCollector,
    eval_env: EnvBase,
    actor: ProbabilisticActor,
    replay_buffer: TensorDictPrioritizedReplayBuffer | TensorDictReplayBuffer,
    loss_module: LossModule,
    optimizers: tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer],
    target_net_updater: SoftUpdate,
    logger: WandbLogger,
    config: ExperimentConfigSchema,
):
    """Trains and evaluate."""
    optimizer_actor, optimizer_critic, optimizer_alpha = optimizers
    collected_frames = 0
    pbar = tqdm.tqdm(total=config.collector.total_frames, unit="frames")
    num_updates = int(1 * config.collector.frames_per_batch * 1.0)
    sampling_start = time.time()
    for i, tensordict in enumerate(collector):
        sampling_time = time.time() - sampling_start
        # Update weights of the inference policy
        collector.update_policy_weights_()
        pbar.update(tensordict.numel())
        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames
        # Optimization steps
        training_start = time.time()
        if collected_frames >= config.collector.init_random_frames:
            losses_values = TensorDict({}, batch_size=[num_updates])
            for i in range(num_updates):
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
                losses_values[i] = loss_td.select(
                    "loss_actor", "loss_qvalue", "loss_alpha"
                ).detach()
                # Update qnet_target params
                target_net_updater.step()
                # Update priority
                if config.replay_buffer.prb:
                    replay_buffer.update_priority(sampled_tensordict)
        training_time = time.time() - training_start
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]
        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= config.collector.init_random_frames:
            metrics_to_log["train/q_loss"] = (
                losses_values.get("loss_qvalue").mean().item()
            )
            metrics_to_log["train/actor_loss"] = (
                losses_values.get("loss_actor").mean().item()
            )
            metrics_to_log["train/alpha_loss"] = (
                losses_values.get("loss_alpha").mean().item()
            )
            metrics_to_log["train/alpha"] = loss_td["alpha"].item()
            metrics_to_log["train/entropy"] = loss_td["entropy"].item()
            metrics_to_log["train/sampling_time"] = sampling_time
            metrics_to_log["train/training_time"] = training_time
        # Evaluation
        if (
            abs(collected_frames % config.evaluation.eval_iter)
            < config.collector.frames_per_batch
        ):
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    max_steps=config.evaluation.eval_rollout_steps,
                    policy=actor,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/time"] = eval_time

                # print evolution of portfolio value
                portfolio_value = utils.compute_portfolio_value(eval_rollout)

                figure = plt.figure()
                plt.plot(portfolio_value)
                utils.log_figure(figure, "portfolio_value")
                plt.close(figure)
        utils.log_metrics(logger, metrics_to_log, collected_frames)
        sampling_start = time.time()
    collector.shutdown()
