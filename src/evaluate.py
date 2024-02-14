"""Module to evaluate an agent."""

import time

import matplotlib.pyplot as plt
import torch
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

import logger
import utils
from config.schemas import EvaluationConfigSchema


def evaluate(
    eval_env: EnvBase,
    actor: ProbabilisticActor,
    config: EvaluationConfigSchema,
):
    metrics_to_log = {}
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        eval_start = time.time()
        eval_rollout = eval_env.rollout(
            max_steps=config.eval_rollout_steps,
            policy=actor,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        eval_time = time.time() - eval_start
        eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
        metrics_to_log["eval/reward"] = eval_reward
        metrics_to_log["timer/eval/time"] = eval_time

        # print evolution of portfolio value
        portfolio_value = utils.compute_portfolio_value(eval_rollout)
        figure = plt.figure()
        plt.plot(portfolio_value)
        logger.log_figure(figure, "portfolio_value")
        plt.close(figure)

        # print evolution of number or shares
        figure = plt.figure()
        num_shares_owned = eval_rollout["next", "num_shares_owned"]
        num_shares_owned = num_shares_owned.squeeze(-1)
        if len(num_shares_owned.shape) == 2:
            num_shares_owned = num_shares_owned.mean(dim=0)
        num_shares_owned = num_shares_owned.tolist()
        figure = plt.figure()
        plt.plot(num_shares_owned)
        logger.log_figure(figure, "num_shares_owned")
        plt.close(figure)
    return metrics_to_log
