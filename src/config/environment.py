"""Module that contains the config classes for the supported environments."""

import pydantic


@pydantic.dataclasses.dataclass
class EnvironmentConfigSchema:
    """Configuration schema for the environment."""

    batch_size: int | None = 1
    cash: float | str = 1e6
    random_initial_distribution: float | None = None
    start_date: str | None = None
    end_date: str | None = None
