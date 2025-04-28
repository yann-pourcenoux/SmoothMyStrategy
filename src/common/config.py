"""Module that contains the config schemas."""

import os
from dataclasses import field

import pydantic
import torch
from hydra.core.config_store import ConfigStore

from environments.config import EnvironmentConfigSchema


@pydantic.dataclasses.dataclass
class DataPreprocessingConfigSchema:
    """Configuration for DataPreprocessing.

    Attributes:
        technical_indicators (list[str]): list of technical indicators to use.
        start_date (str): start date to use for the data.
        end_date (str): end date to use for the data.
    """

    technical_indicators: list[str] = field(default_factory=list)
    start_date: str = "${train_environment.start_date}"
    end_date: str = "${eval_environment.end_date}"


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
class RLAgentConfigSchema:
    """Configuration schema for neural network-based agents."""

    hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (256, 256))
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    activation: str = "relu"
    model_path: str | None = None


@pydantic.dataclasses.dataclass
class QuantAgentConfigSchema:
    """Configuration schema for traditional agents."""

    algorithm_name: str | None = None


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
class BaseExperimentConfigSchema:
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
            random_initial_distribution=1.0,
            start_date="${train_environment.end_date}",
        )
    )
    evaluation: EvaluationConfigSchema = field(default_factory=EvaluationConfigSchema)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@pydantic.dataclasses.dataclass
class RLExperimentConfigSchema(BaseExperimentConfigSchema):
    """Configuration schema for the RL experiment."""

    collector: CollectorConfigSchema = field(default_factory=CollectorConfigSchema)
    replay_buffer: ReplayBufferConfigSchema = field(
        default_factory=ReplayBufferConfigSchema
    )

    agent: RLAgentConfigSchema = field(default_factory=RLAgentConfigSchema)
    loss: LossConfigSchema = field(default_factory=LossConfigSchema)
    optimizer: OptimizerConfigSchema = field(default_factory=OptimizerConfigSchema)

    training: TrainingConfigSchema = field(default_factory=TrainingConfigSchema)


@pydantic.dataclasses.dataclass
class QuantExperimentConfigSchema(BaseExperimentConfigSchema):
    """Configuration schema for the quant experiment."""

    agent: QuantAgentConfigSchema = field(default_factory=QuantAgentConfigSchema)


cs = ConfigStore.instance()

# Create convenience experiment configurations
cs.store(name="rl_experiment", node=RLExperimentConfigSchema())
cs.store(name="quant_experiment", node=QuantExperimentConfigSchema())
