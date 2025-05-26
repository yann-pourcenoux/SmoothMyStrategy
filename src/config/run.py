"""Module that contains the config schemas."""

import os
from dataclasses import field

import pydantic
import torch
from hydra.core.config_store import ConfigStore

from config.base import (
    BaseAgentConfigSchema,
    BasePolicyConfigSchema,
    BaseTrainingConfigSchema,
)
from config.evaluation import EvaluationConfigSchema
from config.quant import QuantAgentConfigSchema, QuantPolicyConfigSchema
from config.rl import RLAgentConfigSchema, RLPolicyConfigSchema, RLTrainingConfigSchema


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
class RunParameters:
    seed: int = 0
    device: str | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@pydantic.dataclasses.dataclass
class BaseTrainingRunConfigSchema:
    """Configuration schema to train a model."""

    training: BaseTrainingConfigSchema = field(default_factory=BaseTrainingConfigSchema)
    logging: LoggingConfigSchema = field(default_factory=LoggingConfigSchema)
    evaluation: EvaluationConfigSchema = field(default_factory=EvaluationConfigSchema)
    run_parameters: RunParameters = field(default_factory=RunParameters)
    agent: BaseAgentConfigSchema = field(default_factory=BaseAgentConfigSchema)


@pydantic.dataclasses.dataclass
class EvaluationRunConfigSchema:
    logging: LoggingConfigSchema = field(default_factory=LoggingConfigSchema)
    evaluation: EvaluationConfigSchema = field(default_factory=EvaluationConfigSchema)
    policy: BasePolicyConfigSchema = field(default_factory=BasePolicyConfigSchema)
    run_parameters: RunParameters = field(default_factory=RunParameters)


@pydantic.dataclasses.dataclass
class CalibrationRunConfigSchema(BaseTrainingConfigSchema):
    """Configuration schema for the quant experiment."""

    agent: QuantAgentConfigSchema = field(default_factory=QuantAgentConfigSchema)


@pydantic.dataclasses.dataclass
class TrainingConfigRunSchema(BaseTrainingRunConfigSchema):
    training: RLTrainingConfigSchema = field(default_factory=RLTrainingConfigSchema)
    agent: RLAgentConfigSchema = field(default_factory=RLAgentConfigSchema)


@pydantic.dataclasses.dataclass
class RLEvaluationRunConfigSchema(EvaluationRunConfigSchema):
    policy: RLPolicyConfigSchema = field(default_factory=RLPolicyConfigSchema)


@pydantic.dataclasses.dataclass
class QuantEvaluationRunConfigSchema(EvaluationRunConfigSchema):
    policy: QuantPolicyConfigSchema = field(default_factory=QuantPolicyConfigSchema)


cs = ConfigStore.instance()

cs.store(name="calibration", node=CalibrationRunConfigSchema())
cs.store(name="training", node=TrainingConfigRunSchema())
cs.store(name="rl_evaluation", node=RLEvaluationRunConfigSchema())
cs.store(name="quant_evaluation", node=QuantEvaluationRunConfigSchema())
