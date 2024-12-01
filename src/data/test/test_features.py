"""Unit tests for src/data/features.py."""

import numpy as np
import pandas as pd
import pytest

from data.features import FeatureGenerator


def _create_test_data() -> pd.DataFrame:
    """Create test data."""
    dates = pd.date_range("2020-01-01", "2020-01-10")
    data = pd.DataFrame(
        {
            "close": [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
            "date": dates,
            "ticker": "AAPL",
        }
    )
    return data


def test_feature_parsing() -> None:
    """Test the feature name parsing."""
    generator = FeatureGenerator()

    # Test log return parsing
    func_name, params = generator._parse_feature("log_return_8")
    assert func_name == "log_return"
    assert params == {"shift": 8}

    # Test simple return parsing
    func_name, params = generator._parse_feature("return_5")
    assert func_name == "return"
    assert params == {"shift": 5}

    # Test invalid feature name
    with pytest.raises(ValueError, match="Unknown feature pattern"):
        generator._parse_feature("invalid_feature_name")


def test_feature_generation() -> None:
    """Test the feature generation."""
    generator = FeatureGenerator()

    data = _create_test_data()

    # Test feature generation
    features = generator.generate_features(data, ["log_return_1", "return_2"])

    assert "log_return_1" in features.columns
    assert "return_2" in features.columns
    assert len(features) == len(data)

    # Test log return values
    expected_log_return = np.log(data["close"] / data["close"].shift(1))
    np.testing.assert_array_equal(features["log_return_1"].values, expected_log_return)

    # Test return values
    expected_return = (data["close"] / data["close"].shift(2)) - 1
    np.testing.assert_array_equal(features["return_2"].values, expected_return)


def test_invalid_feature() -> None:
    """Test handling of invalid feature names."""
    generator = FeatureGenerator()
    data = _create_test_data()

    with pytest.raises(ValueError, match="Unknown feature pattern"):
        generator.generate_features(data, ["invalid_feature"])


def test_missing_price_column() -> None:
    """Test handling of missing price column."""
    generator = FeatureGenerator()
    data = _create_test_data()

    # Rename the close column
    data.rename(columns={"close": "wrong_column"}, inplace=True)
    with pytest.raises(KeyError):
        generator.generate_features(data, ["log_return_1"])
