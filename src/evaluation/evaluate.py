"""Module to evaluate an agent or algorithm."""

import contextlib
import time
from typing import Callable, Dict

import pandas as pd
import quantstats_lumi as qs
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from config.evaluation import EvaluationConfigSchema
from environment.trading import TradingEnv


def rollout(
    eval_env: EnvBase,
    policy: Callable | torch.nn.Module,
    config: EvaluationConfigSchema,
) -> TensorDict:
    """Rollout an environment according to a policy (actor or algorithm wrapper)."""

    # Check if the policy is an RL actor to apply RL-specific logic
    is_rl_actor = isinstance(policy, ProbabilisticActor)

    if is_rl_actor:
        policy.eval()
        exploration_context = set_exploration_type(
            ExplorationType.from_str(config.parameters.exploration_type)
        )
    else:
        # For non-RL policies (like our wrapper), create a null context
        exploration_context = contextlib.nullcontext()

    with (
        exploration_context,
        torch.no_grad(),
        torch.inference_mode(),
    ):
        eval_rollout = eval_env.rollout(
            max_steps=config.parameters.eval_rollout_steps,
            policy=policy,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )

    if is_rl_actor:
        policy.train()

    return eval_rollout


def compute_rl_eval_metrics(eval_rollout: TensorDict) -> Dict[str, float]:
    """Computes the RL related evaluation metrics and returns them."""

    episode_end = eval_rollout["next", "done"]
    episode_rewards = eval_rollout["next", "episode_reward"][episode_end]
    episode_rewards = episode_rewards.mean().item()
    episode_length = eval_rollout["next", "step_count"][episode_end]
    episode_length = episode_length.sum().item() / len(episode_length)

    metrics_to_log = {
        "eval/reward": episode_rewards,
        "eval/episode_length": episode_length,
        "eval/average_reward_per_step": episode_rewards / episode_length,
    }
    return metrics_to_log


def compute_eval_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Analyse the performance from a csv and plot."""
    daily_returns = df["daily_returns"]
    metrics_to_log = {
        "eval/sharpe_ratio": qs.stats.sharpe(daily_returns),
        "eval/calmar_ratio": qs.stats.calmar(daily_returns),
    }

    return metrics_to_log


def evaluate(
    eval_env: TradingEnv,
    policy: Callable | torch.nn.Module,
    config: EvaluationConfigSchema,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Run an evaluation rollout, log metrics and save data for analysis.

    Args:
        eval_env (EnvBase): Environment to evaluate.
        policy (Callable | torch.nn.Module): Policy to evaluate (RL actor or wrapped algorithm).
        config (EvaluationConfigSchema): Evaluation configuration.

    Returns:
        tuple[Dict[str, float], pd.DataFrame]: Metrics to log and the evaluation DataFrame.
    """
    eval_start = time.time()
    eval_rollout = rollout(eval_env, policy, config)
    eval_time = time.time() - eval_start

    eval_df = eval_env.process_rollout(eval_rollout)

    metrics_to_log = {"timer/eval/time": eval_time}
    metrics_to_log.update(compute_rl_eval_metrics(eval_rollout))
    metrics_to_log.update(compute_eval_metrics(eval_df))

    return metrics_to_log, eval_df
