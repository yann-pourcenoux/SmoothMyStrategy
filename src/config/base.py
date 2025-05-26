"""Module that contains the base config schemas."""

from dataclasses import field

import pydantic

from config.data import DataLoaderConfigSchema, DataPreprocessingConfigSchema
from config.environment import EnvironmentConfigSchema


@pydantic.dataclasses.dataclass
class BaseAgentConfigSchema:
    """Something."""


@pydantic.dataclasses.dataclass
class BaseTrainingConfigSchema:
    loading: DataLoaderConfigSchema = field(default_factory=DataLoaderConfigSchema)
    preprocessing: DataPreprocessingConfigSchema = field(
        default_factory=DataPreprocessingConfigSchema
    )

    environment: EnvironmentConfigSchema = field(
        default_factory=lambda: EnvironmentConfigSchema(
            end_date="${eval_environment.start_date}",
        )
    )


@pydantic.dataclasses.dataclass
class BasePolicyConfigSchema:
    """Policy."""
