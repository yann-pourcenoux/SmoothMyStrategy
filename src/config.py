"""Module that contains the config schemas."""

import os
from dataclasses import field

import pydantic
import torch
from hydra.core.config_store import ConfigStore

from data.config import DataLoaderConfigSchema, DataPreprocessingConfigSchema
from environments.config import EnvironmentConfigSchema
from evaluation.config import AnalysisConfigSchema, EvaluationConfigSchema
from quant.config import QuantAgentConfigSchema
from rl.config import (
    CollectorConfigSchema,
    LossConfigSchema,
    OptimizerConfigSchema,
    ReplayBufferConfigSchema,
    RLAgentConfigSchema,
    TrainingConfigSchema,
)


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
