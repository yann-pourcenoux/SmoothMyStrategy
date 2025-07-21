"""Module that contains the functions to add features to the data."""

import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


class FeatureGenerator:
    """Generates features from price data based on feature names.

    This class handles both parsing feature names and generating the corresponding
    features from price data. Feature names encode both the type of feature and its
    parameters (e.g., 'log_return_8' for an 8-period log return).
    """

    FEATURE_PATTERNS = {
        r"^log_return_(\d+)$": ("log_return", ["shift"]),
        r"^return_(\d+)$": ("return", ["shift"]),
        # Add more patterns as needed
    }

    def __init__(self):
        """Initialize the feature generator with available feature functions."""
        self.feature_functions: Dict[str, Callable] = {
            "log_return": self._generate_log_return,
            "return": self._generate_return,
            # Add more feature functions as needed
        }

    def _parse_feature(self, feature_name: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a feature name into function name and parameters.

        Args:
            feature_name: Name of the feature (e.g., 'log_return_8')

        Returns:
            Tuple containing:
                - function name (str)
                - dictionary of parameters

        Raises:
            ValueError: If feature name doesn't match any known pattern
        """
        for pattern, (func_name, param_names) in self.FEATURE_PATTERNS.items():
            match = re.match(pattern, feature_name)
            if match:
                params = [int(x) for x in match.groups()]
                return func_name, dict(zip(param_names, params))

        raise ValueError(f"Unknown feature pattern: {feature_name}")

    def _generate_log_return(self, data: pd.DataFrame, shift: int) -> pd.Series:
        """Generate log return feature with specified shift.

        Args:
            data: DataFrame containing price data
            shift: Number of periods to shift

        Returns:
            Series containing the log returns
        """
        prices = data["close"]
        return np.log(prices / prices.shift(shift))

    def _generate_return(self, data: pd.DataFrame, shift: int) -> pd.Series:
        """Generate simple return feature with specified shift.

        Args:
            data: DataFrame containing price data
            shift: Number of periods to shift

        Returns:
            Series containing the returns
        """
        prices = data["close"]
        return (prices / prices.shift(shift)) - 1

    def generate_features(self, data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Generate all specified features from the input data.

        Args:
            data: DataFrame containing price data
            feature_names: List of feature names to generate

        Returns:
            DataFrame containing all generated features

        Raises:
            ValueError: If an unknown feature is requested
        """
        for feature_name in feature_names:
            func_name, params = self._parse_feature(feature_name)
            if func_name not in self.feature_functions:
                raise ValueError(f"Unknown feature function: {func_name}")

            feature = self.feature_functions[func_name](data, **params)
            data[feature_name] = feature

        return data
