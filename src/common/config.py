"""Module that contains the config schemas."""

import os
from dataclasses import field
from typing import Any

import pydantic
import torch
from hydra.core.config_store import ConfigStore

from environments.config import EnvironmentConfigSchema


@pydantic.dataclasses.dataclass
class DataPreprocessingConfigSchema:
    """Configuration for DataPreprocessing.

    Attributes:
        technical_indicators (list[str]): list of technical indicators to use.
        start_date (str | None): start date to use for the data.
        end_date (str | None): end date to use for the data.
    """

    technical_indicators: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_date: str | None = "${train_environment.start_date}"
    end_date: str | None = "${eval_environment.end_date}"


@pydantic.dataclasses.dataclass
class DataLoaderConfigSchema:
    """Configuration for DataLoader.

    Attributes:
        tickers (list[str]): list of tickers to load data for.
    """

    tickers: list[str] = field(default_factory=list)

    @pydantic.field_validator("tickers")
    @classmethod
    def sort_tickers(cls, tickers: list[str]) -> list[str]:
        """Sort the tickers."""
        return sorted(tickers)


@pydantic.dataclasses.dataclass
class LoggingConfigSchema:
    """Configuration schema for logging."""

    logging_directory: str = "outputs"
    experiment: str | None = None
    project: str = "debug"
    online: bool = True

    def __post_init__(self):
        if self.logging_directory:
            os.makedirs(self.logging_directory, exist_ok=True)


@pydantic.dataclasses.dataclass
class AgentConfigSchema:
    """Configuration schema for the agent."""

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

    def __post_init__(self):
        if self.buffer_scratch_dir is None:
            self.buffer_scratch_dir = "outputs/scratch_replay_buffer"


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
class EvaluationConfigSchema:
    """Configuration schema for the evaluation."""

    eval_rollout_steps: int = 1_000_000
    exploration_type: str = "deterministic"


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


@pydantic.dataclasses.dataclass
class AnalysisConfigSchema:
    """Configuration schema for the analysis."""


@pydantic.dataclasses.dataclass
class ExperimentConfigSchema:
    """Configuration schema to train a model."""

    loading: DataLoaderConfigSchema = field(default_factory=DataLoaderConfigSchema)
    preprocessing: DataPreprocessingConfigSchema = field(
        default_factory=DataPreprocessingConfigSchema
    )

    analysis: AnalysisConfigSchema = field(default_factory=AnalysisConfigSchema)

    logging: LoggingConfigSchema = field(default_factory=LoggingConfigSchema)
    seed: int = 0
    device: str | None = None

    train_environment: EnvironmentConfigSchema = field(
        default_factory=lambda: EnvironmentConfigSchema(
            end_date="${eval_environment.start_date}",
        )
    )
    eval_environment: EnvironmentConfigSchema = field(
        default_factory=lambda: EnvironmentConfigSchema(
            batch_size=1,
            cash="${train_environment.cash}",
            fixed_initial_distribution=True,
            start_date="${train_environment.end_date}",
        )
    )

    collector: CollectorConfigSchema = field(default_factory=CollectorConfigSchema)
    replay_buffer: ReplayBufferConfigSchema = field(
        default_factory=ReplayBufferConfigSchema
    )

    agent: AgentConfigSchema = field(default_factory=AgentConfigSchema)
    loss: LossConfigSchema = field(default_factory=LossConfigSchema)
    optimizer: OptimizerConfigSchema = field(default_factory=OptimizerConfigSchema)

    training: TrainingConfigSchema = field(default_factory=TrainingConfigSchema)
    evaluation: EvaluationConfigSchema = field(default_factory=EvaluationConfigSchema)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


cs = ConfigStore.instance()
cs.store(name="base_experiment", node=ExperimentConfigSchema)
