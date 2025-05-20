"""Module to hold the config for the quant."""

import pydantic

from config.base import BasePolicyConfigSchema


@pydantic.dataclasses.dataclass
class QuantAgentConfigSchema:
    """Configuration schema for traditional agents."""

    algorithm_name: str | None = None


@pydantic.dataclasses.dataclass
class QuantPolicyConfigSchema(BasePolicyConfigSchema):
    """Policy."""

    algorithm_name: str = ""
