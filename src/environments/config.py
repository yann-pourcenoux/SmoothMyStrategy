"""Module that contains the config classes for the supported environments."""

import pydantic


@pydantic.dataclasses.dataclass
class EnvironmentConfigSchema:
    """Configuration schema for the environment."""

    batch_size: int | None = None
    cash: float = 1e6
    fixed_initial_distribution: bool = False
