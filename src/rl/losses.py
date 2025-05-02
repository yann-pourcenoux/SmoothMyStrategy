"""Module to define the loss functions for the training."""

from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss

from config import LossConfigSchema


def make_loss_module(model, config: LossConfigSchema) -> tuple[SACLoss, SoftUpdate]:
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=config.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=config.alpha_init,
    )
    loss_module.make_value_estimator(gamma=config.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=config.target_update_polyak)
    return loss_module, target_net_updater
