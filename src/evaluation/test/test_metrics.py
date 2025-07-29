"""Test suite for metrics.py module.

This module tests the portfolio analysis functions including portfolio value calculation,
daily returns computation, and time-weighted return calculation under various scenarios
including edge cases.
"""

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

from evaluation.metrics import (
    compute_daily_returns,
    compute_portfolio_value,
    compute_time_weighted_return,
)


class TestComputePortfolioValue:
    """Test cases for compute_portfolio_value function."""

    def test_empty_rollout(self):
        """Test that empty DataFrame returns empty Series.

        Verifies that the function handles empty input gracefully and returns
        an empty Series with proper DatetimeIndex.

        Expected Behavior:
            The function should return an empty pandas Series when given an
            empty DataFrame, maintaining the index structure.
        """
        rollout = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=["cash", "deposit"])
        result = compute_portfolio_value(rollout)

        expected = pd.Series([], dtype=object, index=rollout.index, name=None)
        tm.assert_series_equal(result, expected)

    def test_single_ticker_no_shares(self):
        """Test portfolio value equals cash when no shares are held.

        Verifies the basic case where portfolio value calculation reduces to
        cash holdings only, since no stock positions exist.

        Expected Behavior:
            When no shares are owned across all tickers, portfolio value should
            equal the cash amount for each date.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1500, 2000],
                "close_AAPL": [100, 110, 120],
                "num_shares_owned_AAPL": [0, 0, 0],
            },
            index=dates,
        )

        result = compute_portfolio_value(rollout)
        expected = pd.Series([1000, 1500, 2000], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_single_ticker_with_shares(self):
        """Test portfolio value calculation with shares and price.

        Verifies the core portfolio NAV calculation: cash + (shares x price).
        This tests the fundamental value computation for a single-ticker portfolio.

        Expected Behavior:
            Portfolio value should equal cash plus the market value of all
            stock holdings (shares x current price).
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [600, 800, 1000],
                "close_AAPL": [100, 200, 300],
                "num_shares_owned_AAPL": [2, 2, 2],
            },
            index=dates,
        )

        result = compute_portfolio_value(rollout)
        # cash + shares * price: [600 + 2*100, 800 + 2*200, 1000 + 2*300]
        expected = pd.Series([800, 1200, 1600], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_multiple_tickers(self):
        """Test portfolio value calculation with multiple tickers.

        Verifies that the function correctly aggregates holdings across
        multiple stock positions to compute total portfolio value.

        Expected Behavior:
            The function should sum the market value of all ticker positions
            plus cash to determine total portfolio NAV.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 800, 600],
                "close_AAPL": [100, 110, 120],
                "num_shares_owned_AAPL": [0, 2, 2],
                "close_GOOGL": [150, 200, 100],
                "num_shares_owned_GOOGL": [0, 0, 1],
            },
            index=dates,
        )
        result = compute_portfolio_value(rollout)

        # Day 1: 1000 + 0*100 + 0*150 = 1000
        # Day 2: 800 + 2*110 + 0*200 = 1020
        # Day 3: 600 + 2*120 + 1*100 = 940
        expected = pd.Series([1000, 1020, 940], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_negative_cash(self):
        """Test robustness with negative cash values.

        Verifies the function handles scenarios with negative cash positions,
        which could occur in leveraged portfolios or margin accounts.

        Expected Behavior:
            The function should correctly compute portfolio value even when
            cash is negative, as long as total NAV remains meaningful.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [-500, -300, -300],
                "close_AAPL": [100, 110, 120],
                "num_shares_owned_AAPL": [10, 12, 12],
            },
            index=dates,
        )

        result = compute_portfolio_value(rollout)
        # cash + shares * price: [-500 + 10*100, -300 + 12*110, -300 + 12*120]
        expected = pd.Series([500, 1020, 1140], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_missing_columns(self):
        """Test that function works when no tickers are present.

        Verifies graceful handling when the DataFrame contains only basic
        columns (cash, deposit) without any ticker-specific data.

        Expected Behavior:
            When only cash and deposit columns are present,
            portfolio value should equal cash (no stock holdings).
        """
        # DataFrame with no ticker columns - only cash
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1500],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        result = compute_portfolio_value(rollout)
        expected = pd.Series([1000, 1500], index=rollout.index)
        tm.assert_series_equal(result, expected)


class TestComputeDailyReturns:
    """Test cases for compute_daily_returns function."""

    def test_empty_rollout(self):
        """Test that empty DataFrame returns empty Series.

        Verifies graceful handling of edge case where no data is available
        for return calculation.

        Expected Behavior:
            Function should return an empty pandas Series when given empty
            input, preserving the DatetimeIndex structure.
        """
        rollout = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=["cash", "deposit"])
        result = compute_daily_returns(rollout)

        expected = pd.Series([], dtype=object, index=rollout.index, name=None)
        tm.assert_series_equal(result, expected)

    def test_no_deposits_no_changes(self):
        """Test zero returns when portfolio value doesn't change.

        Verifies the baseline scenario where portfolio value remains constant
        and no external cash flows occur.

        Expected Behavior:
            When there are no deposits and portfolio value is constant,
            daily returns should be zero for all periods.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1000, 1000],
                "deposit": [0, 0, 0],
                "close_AAPL": [100, 100, 100],
                "num_shares_owned_AAPL": [0, 0, 0],
            },
            index=dates,
        )

        result = compute_daily_returns(rollout)
        expected = pd.Series([0.0, 0.0, 0.0], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_with_deposits(self):
        """Test correct pre-money value calculation with deposits.

        Verifies the time-weighted return calculation methodology where
        deposits are properly accounted for in the pre-money value adjustment.

        Expected Behavior:
            When deposit changes, deposits should be calculated and
            pre-money value adjusted to isolate investment performance
            from cash flow effects.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 800, 600],
                "deposit": [0, 200, 200],
                "close_AAPL": [100, 110, 120],
                "num_shares_owned_AAPL": [0, 2, 2],
            },
            index=dates,
        )

        result = compute_daily_returns(rollout)

        # Day 1: Portfolio = 1000, no previous day → return = 0
        # Day 2: Portfolio = 1020, deposit = 200, pre_money = 1000 + 200 = 1200
        #        return = (1020 - 1200) / 1200 = -0.15
        # Day 3: Portfolio = 840, deposit = 200, pre_money = 1020 + 200 = 1220
        #        return = (840 - 1220) / 1220 = -0.31...
        expected = pd.Series([0.0, -0.15, (840 - 1220) / 1220], index=rollout.index)

        tm.assert_series_equal(result, expected, check_exact=False, rtol=1e-10)

    def test_single_day(self):
        """Test handling of single-row DataFrame.

        Verifies edge case behavior when only one day of data is available,
        making return calculation impossible due to lack of comparison period.

        Expected Behavior:
            Single day should return zero since there's no previous day
            for comparison in time-weighted return calculation.
        """
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000],
                "deposit": [500],
                "close_AAPL": [100],
                "num_shares_owned_AAPL": [5],
            },
            index=dates,
        )

        result = compute_daily_returns(rollout)
        expected = pd.Series([0.0], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_zero_pre_money(self):
        """Test handling of zero pre-money value.

        Verifies robust handling of mathematical edge case where pre-money
        value is zero, which would cause division by zero in the return formula.

        Expected Behavior:
            When pre_money_value is zero, function should replace inf/NaN
            with appropriate fallback values to maintain numerical stability.
        """
        # Create scenario where previous portfolio value is 0 but there's a deposit
        rollout = pd.DataFrame(
            {
                "cash": [0, 100],
                "deposit": [0, 100],
                "close_AAPL": [100, 100],
                "num_shares_owned_AAPL": [0, 1],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        result = compute_daily_returns(rollout)

        # Day 1: return = 0 (first day)
        # Day 2: Portfolio = 200, deposit = 100, pre_money = 0 + 100 = 100
        #        return = (200 - 100) / 100 = 1.0
        expected = pd.Series([0.0, 1.0], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_negative_returns(self):
        """Test correct calculation when portfolio value decreases.

        Verifies that the function correctly handles scenarios where
        investment performance is negative.

        Expected Behavior:
            Function should correctly calculate negative returns when
            portfolio value decreases relative to pre-money value.
        """
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1000],
                "deposit": [0, 0],
                "close_AAPL": [100, 80],  # Price drops
                "num_shares_owned_AAPL": [10, 10],
            },
            index=dates,
        )

        result = compute_daily_returns(rollout)

        # Day 1: Portfolio = 2000, return = 0 (first day)
        # Day 2: Portfolio = 1800, no deposit, pre_money = 2000
        #        return = (1800 - 2000) / 2000 = -0.1
        expected = pd.Series([0.0, -0.1], index=rollout.index)

        tm.assert_series_equal(result, expected)

    def test_no_deposits_with_price_changes(self):
        """Test daily returns calculation with price changes but no deposits.

        Verifies that the function correctly calculates returns when stock prices
        change but there are no external cash flows (deposits).

        Expected Behavior:
            When there are no deposits (deposit remains constant) but
            stock prices change, daily returns should reflect the pure price
            appreciation/depreciation of the portfolio holdings.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [0, 0, 0],  # No cash to isolate stock returns
                "deposit": [0, 0, 0],  # No deposits
                "close_AAPL": [100, 110, 99],  # Price goes up then down
                "num_shares_owned_AAPL": [10, 10, 10],  # Constant holdings
            },
            index=dates,
        )

        result = compute_daily_returns(rollout)

        # Day 1: Portfolio = 1000, return = 0 (first day)
        # Day 2: Portfolio = 1100, no deposit, pre_money = 1000
        #        return = (1100 - 1000) / 1000 = 0.1
        # Day 3: Portfolio = 990, no deposit, pre_money = 1100
        #        return = (990 - 1100) / 1100 = -0.1
        expected = pd.Series([0.0, 0.1, -0.1], index=rollout.index)

        tm.assert_series_equal(result, expected)


class TestComputeTimeWeightedReturn:
    """Test cases for compute_time_weighted_return function."""

    def test_empty_rollout(self):
        """Test that empty DataFrame returns 0.0.

        Verifies graceful handling of edge case where no return data
        is available for time-weighted return calculation.

        Expected Behavior:
            Function should return 0.0 for empty input, representing
            no return when no investment periods exist.
        """
        rollout = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=["cash", "deposit"])
        result = compute_time_weighted_return(rollout)

        assert result == 0.0

    def test_single_day_no_annualize(self):
        """Test single day returns 0.0 without annualization.

        Verifies baseline behavior when only one day of data exists,
        preventing meaningful return calculation.

        Expected Behavior:
            Single day should return 0.0 since there are no period
            returns to chain in the TWR calculation.
        """
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000],
                "deposit": [500],
                "close_AAPL": [100],
                "num_shares_owned_AAPL": [5],
            },
            index=dates,
        )

        result = compute_time_weighted_return(rollout, annualize=False)

        assert result == 0.0

    def test_price_appreciation_no_annualize(self):
        """Test TWR calculation with price appreciation.

        Expected Behavior:
            Should compute ∏(1 + r_t) - 1 correctly for price appreciation scenario.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1000, 1000],
                "deposit": [0, 0, 0],
                "close_AAPL": [100, 110, 120],
                "num_shares_owned_AAPL": [10, 10, 10],
            },
            index=dates,
        )

        result = compute_time_weighted_return(rollout, annualize=False)
        expected_twr = 0.1

        assert np.isclose(result, expected_twr, rtol=1e-2)

    def test_no_price_changes_no_annualize(self):
        """Test TWR calculation when prices don't change.

        Expected Behavior:
            Should return zero when there are no price changes.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1000, 1000],
                "deposit": [0, 0, 0],
                "close_AAPL": [100, 100, 100],
                "num_shares_owned_AAPL": [10, 10, 10],
            },
            index=dates,
        )

        result = compute_time_weighted_return(rollout, annualize=False)
        expected_twr = 0.0

        assert np.isclose(result, expected_twr, rtol=1e-2)

    def test_declining_prices_no_annualize(self):
        """Test TWR calculation with declining prices.

        Expected Behavior:
            Should compute negative returns correctly for declining price scenario.
        """
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1000, 1000],
                "deposit": [0, 0, 0],
                "close_AAPL": [100, 90, 80],
                "num_shares_owned_AAPL": [10, 10, 10],
            },
            index=dates,
        )

        result = compute_time_weighted_return(rollout, annualize=False)
        expected_twr = -0.1

        assert np.isclose(result, expected_twr, rtol=1e-2)

    def test_annualize(self):
        """Test annualization formula with different periods_per_year.

        Verifies the mathematical correctness of the annualization formula
        and its proper application across different time periods.

        Expected Behavior:
            Should apply correct annualization: (1 + TWR)^(periods_per_year/num_periods) - 1
            to convert period returns into annualized equivalent returns.
        """
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1000],
                "deposit": [0, 0],
                "close_AAPL": [100, 110],  # 5% gain
                "num_shares_owned_AAPL": [10, 10],
            },
            index=dates,
        )

        result = compute_time_weighted_return(rollout, annualize=True, periods_per_year=252)

        # Portfolio values: [2000, 2100]
        # Daily return: [0, 0.05]
        # Raw TWR = (1+0) * (1+0.05) - 1 = 0.05
        # Annualized: (1.05)^(252/2) - 1 = (1.05)^126 - 1
        raw_twr = 0.05
        expected = (1 + raw_twr) ** (252 / 2) - 1

        assert np.isclose(result, expected, rtol=1e-10)

    def test_annualize_one_period(self):
        """Test annualization edge case with single period.

        Verifies robust handling of annualization when only one data point
        exists, which should result in meaningful fallback behavior.

        Expected Behavior:
            Single period with annualization should handle division correctly
            and return zero when no meaningful return calculation is possible.
        """
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000],
                "deposit": [0],
                "close_AAPL": [100],
                "num_shares_owned_AAPL": [10],
            },
            index=dates,
        )

        result = compute_time_weighted_return(rollout, annualize=True, periods_per_year=252)

        # Single day → raw TWR = 0.0
        # Annualized: (1.0)^(252/1) - 1 = 0.0
        assert result == 0.0


class TestEdgeCases:
    """Test edge cases and non-functional requirements."""

    @pytest.mark.parametrize(
        "func", [compute_portfolio_value, compute_daily_returns, compute_time_weighted_return]
    )
    def test_input_not_mutated(self, func):
        """Test that functions do not mutate input DataFrame.

        Verifies that all metrics functions preserve input data integrity
        by not modifying the original DataFrame during computation.

        Args:
            func: Function to test for input mutation. One of the three
                 metrics functions from the module.

        Expected Behavior:
            Input DataFrame should remain completely unchanged after
            function execution, ensuring safe reuse of data.
        """
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1100],
                "deposit": [500, 500],
                "close_AAPL": [100, 110],
                "num_shares_owned_AAPL": [5, 5],
            },
            index=dates,
        )
        original_rollout = rollout.copy()

        # Call function (ignore result)
        if func == compute_time_weighted_return:
            func(rollout, annualize=False)
        else:
            func(rollout)

        # Verify input DataFrame unchanged
        tm.assert_frame_equal(rollout, original_rollout)

    @pytest.mark.parametrize("func", [compute_portfolio_value, compute_daily_returns])
    def test_preserve_datetime_index(self, func):
        """Test that returned Series preserve DatetimeIndex.

        Verifies that all Series-returning functions maintain proper
        temporal indexing for time-series analysis compatibility.

        Args:
            func: Function to test for index preservation. One of the
                 Series-returning metrics functions.

        Expected Behavior:
            Returned Series should have the same DatetimeIndex as the
            input DataFrame, preserving temporal alignment.
        """
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        rollout = pd.DataFrame(
            {
                "cash": [1000, 1100],
                "deposit": [500, 500],
                "close_AAPL": [100, 110],
                "num_shares_owned_AAPL": [5, 5],
            },
            index=dates,
        )

        result = func(rollout)

        # Check that result has same DatetimeIndex
        assert isinstance(result.index, pd.DatetimeIndex)
        tm.assert_index_equal(result.index, rollout.index)
