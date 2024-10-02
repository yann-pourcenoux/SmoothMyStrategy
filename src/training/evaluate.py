"""Module to evaluate an agent."""

import time
from typing import Dict

import pandas as pd
import quantstats_lumi as qs
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from common.config import EvaluationConfigSchema
from environments.trading import TradingEnv


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
        "sharpe_ratio": qs.stats.sharpe(daily_returns),
        "calmar_ratio": qs.stats.calmar(daily_returns),
    }

    return metrics_to_log


def evaluate(
    eval_env: TradingEnv,
    actor: ProbabilisticActor,
    config: EvaluationConfigSchema,
) -> Dict[str, float]:
    """Run an evaluation rollout, log metrics and save data for analysis.

    Args:
        eval_env (EnvBase): Environment to evaluate.
        actor (ProbabilisticActor): Actor to evaluate.
        config (EvaluationConfigSchema): Evaluation configuration.

    Returns:
        Dict[str, float]: Metrics to log.
    """
    eval_start = time.time()
    eval_rollout = rollout(eval_env, actor, config)
    eval_time = time.time() - eval_start

    eval_df = eval_env.process_rollout(eval_rollout)

    metrics_to_log = {"timer/eval/time": eval_time}
    metrics_to_log.update(compute_rl_eval_metrics(eval_rollout))
    metrics_to_log.update(compute_eval_metrics(eval_df))

    return metrics_to_log, eval_df
