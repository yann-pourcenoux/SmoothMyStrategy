"""Module to define the utilities for the investment environment."""

from torchrl.envs import CatTensors, Compose, DoubleToFloat, EnvBase, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, VecNormV2

from config.environment import EnvironmentConfigSchema
from data.container import DataContainer
from environment.base import BaseTradingEnv
from environment.signal_action import SignalActionEnv
from environment.weight_action import WeightActionEnv


def create_signal_action_env(
    data_container: DataContainer,
    seed: int | None,
    device: str,
    **config_kwargs,
) -> SignalActionEnv:
    """Factory function to create a SignalActionEnv instance from config parameters.

    This function is designed to work with Hydra's instantiate() by accepting
    the environment configuration as keyword arguments.

    Args:
        data_container: DataContainer instance for the environment.
        seed: Random seed for the environment.
        device: Device to run the environment on.
        **config_kwargs: Configuration parameters for EnvironmentConfigSchema.

    Returns:
        SignalActionEnv: Configured signal action environment instance.
    """
    config = EnvironmentConfigSchema(**config_kwargs)
    return SignalActionEnv(
        config=config,
        data_container=data_container,
        seed=seed,
        device=device,
    )


def create_weight_action_env(
    data_container: DataContainer,
    seed: int | None,
    device: str,
    **config_kwargs,
) -> WeightActionEnv:
    """Factory function to create a WeightActionEnv instance from config parameters.

    This function is designed to work with Hydra's instantiate() by accepting
    the environment configuration as keyword arguments.

    Args:
        data_container: DataContainer instance for the environment.
        seed: Random seed for the environment.
        device: Device to run the environment on.
        **config_kwargs: Configuration parameters for EnvironmentConfigSchema.

    Returns:
        WeightActionEnv: Configured weight-based action environment instance.
    """
    config = EnvironmentConfigSchema(**config_kwargs)
    return WeightActionEnv(
        config=config,
        data_container=data_container,
        seed=seed,
        device=device,
    )


def _apply_transform_observation(env: BaseTradingEnv) -> TransformedEnv:
    """Concatenates the columns into an observation key."""
    transformed_env = TransformedEnv(
        env=env,
        transform=Compose(
            CatTensors(
                in_keys=env.technical_indicators,
                dim=-1,
                out_key="observation",
                del_keys=False,
            ),
            VecNormV2(in_keys=["observation"]),
        ),
        device=env.device,
    )
    return transformed_env


def _apply_transforms(env: EnvBase) -> TransformedEnv:
    """Apply the necessary transforms to train using SAC."""
    transformed_env = TransformedEnv(
        env=env,
        transform=Compose(
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),
            RewardSum(),
        ),
        device=env.device,
    )
    return transformed_env


def apply_transforms(env: EnvBase) -> TransformedEnv:
    """Get the environment to train using SAC."""
    env = _apply_transform_observation(env)
    env = _apply_transforms(env)
    return env
