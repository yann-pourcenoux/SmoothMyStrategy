"""Module to hold the config for the evaluation."""

from dataclasses import field

import pydantic

from config.data import DataLoaderConfigSchema, DataPreprocessingConfigSchema
from config.environment import EnvironmentConfigSchema


@pydantic.dataclasses.dataclass
class EvalParametersConfigSchema:
    """Configuration schema for the evaluation."""

    eval_rollout_steps: int = 1_000_000
    exploration_type: str = "deterministic"


@pydantic.dataclasses.dataclass
class AnalysisConfigSchema:
    """Configuration schema for the analysis."""


@pydantic.dataclasses.dataclass
class EvaluationConfigSchema:
    loading: DataLoaderConfigSchema = field(default_factory=DataLoaderConfigSchema)
    preprocessing: DataPreprocessingConfigSchema = field(
        default_factory=DataPreprocessingConfigSchema
    )
    environment: EnvironmentConfigSchema = field(
        default_factory=lambda: EnvironmentConfigSchema(
            batch_size=1,
            cash="${train_environment.cash}",
            random_initial_distribution=1.0,
            start_date="${train_environment.end_date}",
        )
    )
    analysis: AnalysisConfigSchema = field(default_factory=AnalysisConfigSchema)
    parameters: EvalParametersConfigSchema = field(default_factory=EvalParametersConfigSchema)
