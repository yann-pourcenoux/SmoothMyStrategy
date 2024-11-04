"""Unit tests for src/data/features.py."""

from functools import partial
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from data.features import get_feature, log_return, macd


def test_log_return() -> None:
    """Test the log_return function.

    Ensures that the log return is correctly calculated and shifted as specified.
    """
    data = {"adj_close": [100, 102, 101, 103, 104]}
    df = pd.DataFrame(data)
    result = log_return(df.copy(), column="adj_close", shift=1)
    expected = df.copy()
    expected["log_return_1"] = np.log(
        expected["adj_close"] / expected["adj_close"].shift(1)
    )
    expected["log_return_1"] = expected["log_return_1"].shift(1)
    pd.testing.assert_frame_equal(result, expected)


def test_macd() -> None:
    """Test the macd function.

    Verifies that the MACD is correctly calculated using EMA with the specified shift.
    """
    np.random.seed(0)
    data = {"adj_close": np.random.rand(100)}
    df = pd.DataFrame(data)
    result = macd(df.copy(), column="adj_close", shift=26)
    ema_12 = df["adj_close"].ewm(span=12, adjust=False).mean() / df["adj_close"].shift(
        26
    )
    ema_26 = df["adj_close"].ewm(span=26, adjust=False).mean() / df["adj_close"].shift(
        26
    )
    expected_macd = ema_12 - ema_26
    expected = df.copy()
    expected["macd_26"] = expected_macd
    pd.testing.assert_frame_equal(result, expected)


def test_get_feature_log_ret() -> None:
    """Test get_feature for log-return.

    Ensures that the correct log_return function is returned when requested.
    """
    feature_params: Dict[str, Any] = {"column": "adj_close", "shift": 0}
    feature = get_feature("log_return", feature_params)
    assert isinstance(feature, partial)
    assert feature.func == log_return
    assert feature.keywords.get("column") == "adj_close"
    assert feature.keywords.get("shift") == 0


def test_get_feature_macd() -> None:
    """Test get_feature for MACD.

    Ensures that the correct macd function is returned when requested.
    """
    feature_params: Dict[str, Any] = {"column": "adj_close", "shift": 26}
    feature = get_feature("macd", feature_params)
    assert isinstance(feature, partial)
    assert feature.func == macd
    assert feature.keywords.get("column") == "adj_close"
    assert feature.keywords.get("shift") == 26


def test_get_feature_invalid() -> None:
    """Test get_feature with an invalid feature name.

    Verifies that a ValueError is raised when an unknown feature name is provided.
    """
    feature_params: Dict[str, Any] = {"column": "adj_close", "shift": 0}
    with pytest.raises(ValueError):
        get_feature("invalid-feature", feature_params)
