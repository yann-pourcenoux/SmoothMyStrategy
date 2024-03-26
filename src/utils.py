"""Module that holds all the utils function to run the training."""

import time

import torch
import tqdm
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor

from config.schemas import CollectorConfigSchema, ReplayBufferConfigSchema


def make_collector(
    train_env: EnvBase,
    actor_model_explore: ProbabilisticActor,
    config: CollectorConfigSchema,
    device: str = "cpu",
    seed: int = 0,
) -> SyncDataCollector:
    """Make collector."""
    # Set actor to eval mode
    actor_model_explore.eval()

    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        storing_device="cpu",
        env_device=device,
        policy_device=device,
    )
    collector.set_seed(seed)

    # Set actor back to train mode
    actor_model_explore.train()
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


def compute_portfolio_value(tensordict: TensorDict) -> list[float] | list[list[float]]:
    """Compute the value of a portfolio over the time steps and batch dimensions."""
    portfolio_value = tensordict["cash"] + torch.sum(
        tensordict["num_shares_owned"] * tensordict["close"],
        dim=-1,
        keepdim=True,
    )
    portfolio_value = portfolio_value.squeeze(-1)
    if len(portfolio_value.shape) == 2:
        portfolio_value = portfolio_value.mean(dim=0)
    return portfolio_value.tolist()


def get_device(device: str | None) -> torch.device:
    """Get the device."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def collect_data(
    collector: SyncDataCollector,
    replay_buffer: TensorDictPrioritizedReplayBuffer | TensorDictReplayBuffer,
    num_steps_per_episode: int,
    advantage_module: str = "",
) -> dict[str, float]:
    # Set the policy in eval mode
    collector.policy.eval()

    sampling_start = time.time()
    for i, tensordict in tqdm.tqdm(
        enumerate(collector),
        total=num_steps_per_episode,
        desc="Sampling",
        unit="step",
        leave=False,
    ):
        # Stop the loop if reached the number of steps
        if i == num_steps_per_episode:
            break

        # Update weights of the inference policy
        collector.update_policy_weights_()

        # Compute the advantage
        if advantage_module:
            with torch.no_grad():
                tensordict = advantage_module(tensordict)

        # Add to replay buffer
        tensordict = tensordict.reshape(-1)
        replay_buffer.extend(tensordict.cpu())

        # Add metrics
        episode_end = (
            tensordict["next", "done"] if tensordict["next", "done"].any() else False
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

    sampling_time = time.time() - sampling_start
    metrics_to_log["timer/train/sampling_time"] = sampling_time

    # Set the policy back to train mode
    collector.policy.train()
    return metrics_to_log
