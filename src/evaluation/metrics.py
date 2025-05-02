"""Module to analyse the performance of a model."""

import pandas as pd


def compute_daily_returns(portfolio_value: pd.Series) -> pd.Series:
    daily_returns = portfolio_value.pct_change().fillna(0)
    dates = pd.date_range(start="2020-01-01", periods=len(daily_returns), freq="B")
    daily_returns.index = dates
    return daily_returns
