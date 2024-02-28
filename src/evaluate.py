"""Module to evaluate an agent."""

import time

import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from config.schemas import EvaluationConfigSchema


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
):
    metrics_to_log = {}
    eval_start = time.time()
    eval_rollout = rollout(eval_env, actor, config)
    eval_time = time.time() - eval_start

    eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
    metrics_to_log["eval/reward"] = eval_reward
    metrics_to_log["timer/eval/time"] = eval_time

    save_traj(eval_rollout, config.output_path)
    return metrics_to_log


def save_traj(eval_rollout: TensorDict, output_path: str) -> None:
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

    pd.DataFrame.from_dict(columns).to_csv(output_path, index=False)
