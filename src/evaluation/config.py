"""Module to hold the config for the evaluation."""

import pydantic


@pydantic.dataclasses.dataclass
class EvaluationConfigSchema:
    """Configuration schema for the evaluation."""

    eval_rollout_steps: int = 1_000_000
    exploration_type: str = "deterministic"


@pydantic.dataclasses.dataclass
class AnalysisConfigSchema:
    """Configuration schema for the analysis."""
