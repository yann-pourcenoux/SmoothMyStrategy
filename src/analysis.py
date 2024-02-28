"""Module to analyse the performance of a model."""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# import quantstats as qs


def add_portfolio_value(df: pd.DataFrame) -> None:
    close_cols = df.filter(regex="close_*")
    num_shares_cols = df.filter(regex="shares_*")

    # Add the values columns
    for i, value in enumerate(np.transpose(close_cols.values * num_shares_cols.values)):
        df[f"value_{i}"] = value

    df["portfolio_value"] = df["cash"].values + np.sum(
        df.filter(regex="value_*").values, axis=-1
    )


def plot_evolutions(df: pd.DataFrame) -> go.Figure:
    """Plot the evolution of the stocks and the portfolio value."""
    col_names = [col_name for col_name in df.columns if "close_" in col_name]
    col_names.append("portfolio_value")

    df = df[col_names]
    for col_name in col_names:
        df.loc[:, col_name] = df[col_name] / df.at[0, col_name]

    fig = px.line(df)
    return fig


def plot_actions(df: pd.DataFrame) -> go.Figure:
    """Plot the evolution of the stocks and the portfolio value."""
    col_names = [col_name for col_name in df.columns if "action_" in col_name]
    df = df[col_names]
    fig = px.line(df)
    return fig


def plot_portfolio_composition(df) -> go.Figure:
    """Plot the portfolio composition."""
    col_names = [col_name for col_name in df.columns if "value_" in col_name]
    col_names.append("cash")
    portfolio_value = df["portfolio_value"]

    df = df[col_names]
    for col_name in col_names:
        df.loc[:, col_name] = df[col_name] / portfolio_value

    fig = px.area(df)
    return fig


def compute_daily_returns(portfolio_value: pd.Series) -> pd.Series:
    daily_returns = portfolio_value.pct_change().fillna(0)
    dates = pd.date_range(start="2020-01-01", periods=len(daily_returns), freq="B")
    daily_returns.index = dates
    return daily_returns


def analyse(path_to_csv: str) -> Dict[str, float]:
    """Analyse the performance from a csv and plot."""
    df = pd.read_csv(path_to_csv)
    add_portfolio_value(df)

    # daily_returns = compute_daily_returns(df["portfolio_value"])
    # qs.reports.html(returns=daily_returns, output='report.html')

    # fig = plot_evolutions(df)
    # fig.show()

    # fig = plot_portfolio_composition(df)
    # fig.show()

    # fig = plot_actions(df)
    # fig.show()

    return {"final_value": df["portfolio_value"].iloc[-1]}
