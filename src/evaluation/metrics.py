"""Module to analyse the performance of a model."""

import numpy as np
import pandas as pd


def compute_portfolio_value(rollout: pd.DataFrame) -> pd.Series:
    """Calculate portfolio Net Asset Value (NAV) for every date in *rollout*.

    Args:
        rollout (pandas.DataFrame): DataFrame produced by BaseTradingEnv.process_rollout.

    Returns:
        pandas.Series: Portfolio value indexed by date.
    """
    # Get cash amount
    cash = rollout["cash"]

    # Extract tickers from close columns (format: close_TICKER)
    close_cols = [col for col in rollout.columns if col.startswith("close_")]
    tickers = [col.replace("close_", "") for col in close_cols]

    # Calculate total value of stock positions
    stock_value = pd.Series(0, index=rollout.index)

    for ticker in tickers:
        shares = rollout[f"num_shares_owned_{ticker}"]
        price = rollout[f"close_{ticker}"]
        stock_value += shares * price

    # Total portfolio value
    portfolio_value = cash + stock_value

    return portfolio_value


def compute_daily_returns(rollout: pd.DataFrame) -> pd.Series:
    """Return the series of daily Time-Weighted Returns (TWR sub-period returns).

    Args:
        rollout (pandas.DataFrame): DataFrame produced by BaseTradingEnv.process_rollout.

    Returns:
        pandas.Series: Daily time-weighted returns indexed by date.
    """
    # Get portfolio value
    portfolio_value = compute_portfolio_value(rollout)

    # Calculate pre-money values (portfolio value before deposits)
    pre_money_value = portfolio_value.shift(1) + rollout["deposit"]

    # Calculate daily time-weighted returns
    # r_t = (V_t - V_t_pre) / V_t_pre  where V_t_pre is the pre-money value
    daily_returns = (portfolio_value - pre_money_value) / pre_money_value

    # Handle edge cases: when pre_money_value is 0 or very small
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

    return daily_returns


def compute_time_weighted_return(
    rollout: pd.DataFrame, *, annualize: bool = False, periods_per_year: int = 252
) -> float:
    """Compute the Time-Weighted Return (TWR) for the entire period.

    Args:
        rollout (pandas.DataFrame): DataFrame produced by BaseTradingEnv.process_rollout.
        annualize (bool): If True the function converts the raw period TWR into an
            **annualised** figure using ``periods_per_year``. Defaults to False.
        periods_per_year (int): Number of return observations that constitute one year.
            For business-day data the conventional value is 252; for calendar daily data you may
            prefer 365. Ignored when annualize is False.

    Returns:
        float: Raw or annualised time-weighted return, depending on annualize.
    """
    # Get daily time-weighted returns
    daily_returns = compute_daily_returns(rollout)

    # Chain the returns: TWR = ∏(1 + r_t) - 1
    if len(daily_returns) > 0:
        twr = np.prod(1 + daily_returns) - 1
    else:
        twr = 0.0

    if annualize and len(daily_returns) > 0:
        num_periods = len(daily_returns)
        twr = (1 + twr) ** (periods_per_year / num_periods) - 1

    return twr
