"""Module to evaluate an agent."""

import time

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

import logger
import utils
from config.schemas import EvaluationConfigSchema


def rollout(
    eval_env: EnvBase,
    actor: ProbabilisticActor,
    config: EvaluationConfigSchema,
) -> TensorDict:
    """Rollout an environment according to an actor."""
    actor.eval()
    with set_exploration_type(
        ExplorationType.MODE
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

    save_traj(eval_rollout)
    return metrics_to_log


def save_traj(
    eval_rollout: TensorDict,
) -> None:
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

    # print number or shares bought/sold
    figure = plt.figure()
    num_shares_owned = eval_rollout["action"]
    num_shares_owned = num_shares_owned.squeeze(-1)
    if len(num_shares_owned.shape) == 2:
        num_shares_owned = num_shares_owned.mean(dim=0)
    num_shares_owned = num_shares_owned.tolist()
    figure = plt.figure()
    plt.plot(num_shares_owned)
    logger.log_figure(figure, "bought_sold_shares")
    plt.close(figure)

    # print evolution of portfolio value
    portfolio_value = utils.compute_portfolio_value(eval_rollout)
    portfolio_value = [value / 1e6 for value in portfolio_value]
    figure = plt.figure()
    plt.plot(portfolio_value, label="portfolio_value")
    # logger.log_figure(figure, "portfolio_value")
    # plt.close(figure)

    # print evolution of the close price
    # figure = plt.figure()
    close_price = eval_rollout["next", "close"]
    close_price = close_price.squeeze(0, -1)
    close_price = close_price / close_price[0]
    close_price = close_price.tolist()
    plt.plot(close_price, label="close_price")
    plt.legend()
    logger.log_figure(figure, "close_price")
    plt.close(figure)
