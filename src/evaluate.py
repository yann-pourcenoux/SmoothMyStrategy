"""Module to evaluate an agent."""

import os
import time

import pandas as pd
import torch
import wandb
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from config.schemas import EvaluationConfigSchema
from constants import EVALUATION_LOGS_FILENAME


def rollout(
    eval_env: EnvBase,
    actor: ProbabilisticActor,
    config: EvaluationConfigSchema,
) -> TensorDict:
    """Rollout an environment according to an actor."""
    actor.eval()
    with set_exploration_type(
        ExplorationType.from_str(config.exploration_type)
    ), torch.no_grad(), torch.inference_mode():
        eval_rollout = eval_env.rollout(
            max_steps=config.eval_rollout_steps,
            policy=actor,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
    actor.train()
    return eval_rollout


def evaluate(
    eval_env: EnvBase,
    actor: ProbabilisticActor,
    config: EvaluationConfigSchema,
) -> dict[str, float]:
    """Run an evaluation rollout, log metrics and save data dor analysis.

    Args:
        eval_env (EnvBase): Environment to evaluate.
        actor (ProbabilisticActor): Actor to evaluate.
        config (EvaluationConfigSchema): Evaluation configuration.

    Returns:
        dict[str, float]: Metrics to log.
    """
    metrics_to_log = {}
    eval_start = time.time()
    eval_rollout = rollout(eval_env, actor, config)
    eval_time = time.time() - eval_start

    episode_end = eval_rollout["next", "done"]
    episode_rewards = eval_rollout["next", "episode_reward"][episode_end]
    episode_rewards = episode_rewards.mean().item()
    episode_length = eval_rollout["next", "step_count"][episode_end]
    episode_length = episode_length.sum().item() / len(episode_length)

    metrics_to_log["eval/reward"] = episode_rewards
    metrics_to_log["eval/episode_length"] = episode_length
    metrics_to_log["eval/average_reward_per_step"] = episode_rewards / episode_length
    metrics_to_log["timer/eval/time"] = eval_time

    save_traj(eval_rollout)
    return metrics_to_log


def save_traj(eval_rollout: TensorDict) -> None:
    """Save trajectory to file."""

    num_shares_owned = eval_rollout["num_shares_owned"]
    actions = eval_rollout["action"]
    close_prices = eval_rollout["close"]
    # Cash amount has a last dimension of 1 that is unused here after.
    cash = eval_rollout["cash"][..., 0]

    # Select only the first axis
    if eval_rollout.batch_size[0] != 1:
        raise ValueError(
            "Only a batch size of 1 is supported in the trajectory saving."
        )
    num_shares_owned = num_shares_owned[0]
    actions = actions[0]
    close_prices = close_prices[0]
    cash = cash[0]

    columns = {"cash": cash.cpu().numpy()}
    for ticker_idx, (close_price, action, shares_ticker) in enumerate(
        zip(
            torch.unbind(close_prices, dim=-1),
            torch.unbind(actions, dim=-1),
            torch.unbind(num_shares_owned, dim=-1),
        )
    ):
        columns[f"close_{ticker_idx}"] = close_price.cpu().numpy()
        columns[f"action_{ticker_idx}"] = action.cpu().numpy()
        columns[f"shares_{ticker_idx}"] = shares_ticker.cpu().numpy()

    output_path = os.path.join(wandb.run.dir, EVALUATION_LOGS_FILENAME)
    pd.DataFrame.from_dict(columns).to_csv(output_path, index=False)
