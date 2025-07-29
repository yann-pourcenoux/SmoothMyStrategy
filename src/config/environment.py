"""Module that contains the config classes for the supported environments."""

import pydantic


@pydantic.dataclasses.dataclass
class EnvironmentConfigSchema:
    """Configuration schema for the environment."""

    _target_: str = "environment.utils.create_signal_action_env"
    batch_size: int | None = 1
    cash: float | str = 1e6
    monthly_cash: float | str = 0.0
    random_initial_distribution: float | None = None
    start_date: str | None = None
    end_date: str | None = None
