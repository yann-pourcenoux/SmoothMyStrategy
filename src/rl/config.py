"""Module to hold the config for the RL."""

from dataclasses import field

import pydantic


@pydantic.dataclasses.dataclass
class RLAgentConfigSchema:
    """Configuration schema for neural network-based agents."""

    hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (256, 256))
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    activation: str = "relu"
    model_path: str | None = None


@pydantic.dataclasses.dataclass
class LossConfigSchema:
    """Configuration schema for the loss."""

    loss_function: str = "l2"
    alpha_init: float = 1.0
    gamma: float = 0.99
    target_update_polyak: float = 0.995


@pydantic.dataclasses.dataclass
class ReplayBufferConfigSchema:
    """Configuration schema for the replay buffer."""

    batch_size: int = 512
    prb: bool = False
    alpha: float = 0.7
    beta: float = 0.5
    buffer_size: int = 10000000
    buffer_scratch_dir: str | None = None
    prefetch: int = 3
    pin_memory: bool = False


@pydantic.dataclasses.dataclass
class OptimizerConfigSchema:
    """Configuration schema for the optimizer."""

    actor_lr: float = 3.0e-4
    critic_lr: float = "${optimizer.actor_lr}"
    alpha_lr: float = "${optimizer.actor_lr}"

    weight_decay: float = 0.0
    adam_eps: float = 1.0e-8

    T_max: int = 100000
    eta_min: float = 0.0


@pydantic.dataclasses.dataclass
class TrainingConfigSchema:
    """Configuration schema for the training."""

    num_epochs: int = 100
    num_steps_per_epoch: int = 1000


@pydantic.dataclasses.dataclass
class CollectorConfigSchema:
    """Configuration schema for the collector."""

    total_frames: int = -1
    frames_per_batch: int = 64
    storage_device: str = "cpu"
