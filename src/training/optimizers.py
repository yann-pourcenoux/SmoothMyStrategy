"""Module to define optimizers to train agents."""

from torch import optim
from torchrl.objectives import LossModule

from common.config import OptimizerConfigSchema


def make_sac_optimizer(
    loss_module: LossModule, config: OptimizerConfigSchema
) -> tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=config.actor_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=config.critic_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=config.alpha_lr,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha
