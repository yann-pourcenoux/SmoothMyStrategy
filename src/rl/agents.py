"""Module to create agents to be trained using reinforcement learning."""

import torch
from models.models import get_activation
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal

from config import RLAgentConfigSchema


def make_sac_agent(
    train_env: EnvBase,
    eval_env: EnvBase,
    config: RLAgentConfigSchema,
    device,
) -> tuple[nn.ModuleList, ProbabilisticActor]:
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": config.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(config.activation),
        "norm_class": torch.nn.BatchNorm1d,
        "norm_kwargs": {
            "num_features": config.hidden_sizes[0]
        },  # TODO: HARDCODED BECAUSE OF THE HIDDEN SIZES
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{config.default_policy_scale}",
        scale_lb=config.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": config.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(config.activation),
        "norm_class": torch.nn.BatchNorm1d,
        "norm_kwargs": {
            "num_features": config.hidden_sizes[0]
        },  # TODO: HARDCODED BECAUSE OF THE HIDDEN SIZES
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    model.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()
    model.train()

    return model, model[0]


def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params
