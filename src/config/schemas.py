"""Module that contains the config schemas."""

from dataclasses import field

import pydantic
from hydra.core.config_store import ConfigStore


@pydantic.dataclasses.dataclass
class DataPreprocessingConfigSchema:
    """Configuration for DataPreprocessing.

    Attributes:
        technical_indicators (list[str]): list of technical indicators to use.
    """

    technical_indicators: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None


@pydantic.dataclasses.dataclass
class DataLoaderConfigSchema:
    """Configuration for DataLoader.

    Attributes:
        tickers (list[str]): list of tickers to load data for.
    """

    tickers: list[str] = field(default_factory=list)


@pydantic.dataclasses.dataclass
class RestConfigSchema:
    """Configuration fields for all the rest just to test."""

    seed: int = 0
    device: str | None = None


@pydantic.dataclasses.dataclass
class LoggingConfigSchema:
    """Configuration schema for logging."""

    logging_directory: str = "logging"
    experiment: str | None = None
    project: str = "debug"
    online: bool = True


@pydantic.dataclasses.dataclass
class EnvironmentConfigSchema:
    """Configuration schema for the environment."""

    batch_size: int | None = None
    cash_amount: float = 1e6
    max_episode_steps: int = 1000000


@pydantic.dataclasses.dataclass
class AgentConfigSchema:
    """Configuration schema for the agent."""

    hidden_sizes: tuple[int, int] = (256, 256)
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

    learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    adam_eps: float = 1.0e-8
    alpha_lr: float = 3.0e-4


@pydantic.dataclasses.dataclass
class EvaluationConfigSchema:
    """Configuration schema for the evaluation."""

    eval_iter: int = 25000
    eval_rollout_steps: int = 1000


@pydantic.dataclasses.dataclass
class TrainingConfigSchema:
    """Configuration schema for the training."""

    num_epochs: int = 100
    num_steps_per_epoch: int = 1000
    frames_per_batch: int = 1000


@pydantic.dataclasses.dataclass
class CollectorConfigSchema:
    """Configuration schema for the collector."""

    total_frames: int = -1
    frames_per_batch: int = 1280


@pydantic.dataclasses.dataclass
class ExperimentConfigSchema:
    """Configuration schema to train a model."""

    agent: AgentConfigSchema = AgentConfigSchema()
    collector: CollectorConfigSchema = CollectorConfigSchema()
    environment: EnvironmentConfigSchema = EnvironmentConfigSchema()
    evaluation: EvaluationConfigSchema = EvaluationConfigSchema()
    loading: DataLoaderConfigSchema = DataLoaderConfigSchema()
    logging: LoggingConfigSchema = LoggingConfigSchema()
    loss: LossConfigSchema = LossConfigSchema()
    optimizer: OptimizerConfigSchema = OptimizerConfigSchema()
    preprocessing: DataPreprocessingConfigSchema = DataPreprocessingConfigSchema()
    replay_buffer: ReplayBufferConfigSchema = ReplayBufferConfigSchema()
    rest: RestConfigSchema = RestConfigSchema()
    training: TrainingConfigSchema = TrainingConfigSchema()


cs = ConfigStore.instance()
cs.store(name="base_experiment", node=ExperimentConfigSchema)
