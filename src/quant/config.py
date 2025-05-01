"""Module to hold the config for the quant."""

import pydantic


@pydantic.dataclasses.dataclass
class QuantAgentConfigSchema:
    """Configuration schema for traditional agents."""

    algorithm_name: str | None = None
