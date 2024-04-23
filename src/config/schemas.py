"""Module that contains the config schemas."""

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
    """

    technical_indicators: list[str] = field(default_factory=list)
    start_date: str | None = "${train_environment.start_date}"
    end_date: str | None = "${eval_environment.end_date}"


@pydantic.dataclasses.dataclass
class DataLoaderConfigSchema:
    """Configuration for DataLoader.

    Attributes:
        tickers (list[str]): list of tickers to load data for.
    """

    tickers: list[str] = field(default_factory=list)


@pydantic.dataclasses.dataclass
class LoggingConfigSchema:
    """Configuration schema for logging."""

    logging_directory: str = "logging"
    experiment: str | None = None
    project: str = "debug"
    online: bool = True


@pydantic.dataclasses.dataclass
class AgentConfigSchema:
    """Configuration schema for the agent."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    activation: str = "relu"


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

    batch_size: int = 256
    prb: bool = False
    alpha: float = 0.7
    beta: float = 0.5
    buffer_size: int = 1000000
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


@pydantic.dataclasses.dataclass
class EvaluationConfigSchema:
    """Configuration schema for the evaluation."""

    eval_rollout_steps: int = 1_000_000
    exploration_type: str = "mode"


@pydantic.dataclasses.dataclass
class TrainingConfigSchema:
    """Configuration schema for the training."""

    num_epochs: int = 100
    num_steps_per_epoch: int = 1000


@pydantic.dataclasses.dataclass
class CollectorConfigSchema:
    """Configuration schema for the collector."""

    total_frames: int = -1
    frames_per_batch: int = 1024
    storage_device: str = "cpu"


@pydantic.dataclasses.dataclass
class AnalysisConfigSchema:
    """Configuration schema for the analysis."""


@pydantic.dataclasses.dataclass
class ExperimentConfigSchema:
    """Configuration schema to train a model."""

    loading: DataLoaderConfigSchema = DataLoaderConfigSchema()
    preprocessing: DataPreprocessingConfigSchema = DataPreprocessingConfigSchema()

    train_environment: EnvironmentConfigSchema = EnvironmentConfigSchema(
        end_date="${eval_environment.start_date}",
    )
    eval_environment: EnvironmentConfigSchema = EnvironmentConfigSchema(
        batch_size=1,
        fixed_initial_distribution=True,
        start_date="${train_environment.end_date}",
    )

    collector: CollectorConfigSchema = CollectorConfigSchema()
    replay_buffer: ReplayBufferConfigSchema = ReplayBufferConfigSchema()

    agent: AgentConfigSchema = AgentConfigSchema()
    loss: LossConfigSchema = LossConfigSchema()
    optimizer: OptimizerConfigSchema = OptimizerConfigSchema()

    training: TrainingConfigSchema = TrainingConfigSchema()
    evaluation: EvaluationConfigSchema = EvaluationConfigSchema()
    analysis: AnalysisConfigSchema | None = AnalysisConfigSchema()

    logging: LoggingConfigSchema = LoggingConfigSchema()
    seed: int = 0
    device: str | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


cs = ConfigStore.instance()
cs.store(name="base_experiment", node=ExperimentConfigSchema)
