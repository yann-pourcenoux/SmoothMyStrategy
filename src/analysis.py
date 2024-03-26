"""Module to analyse the performance of a model."""

import os
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import quantstats as qs
import wandb

from constants import EVALUATION_LOGS_FILENAME, METRICS_REPORT_FILENAME


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


def analyse() -> Dict[str, float]:
    """Analyse the performance from a csv and plot."""
    df = pd.read_csv(os.path.join(wandb.run.dir, EVALUATION_LOGS_FILENAME))
    add_portfolio_value(df)

    metrics_to_log = {"final_value": df["portfolio_value"].iloc[-1]}

    daily_returns = compute_daily_returns(df["portfolio_value"])

    report_path = os.path.join(wandb.run.dir, METRICS_REPORT_FILENAME)
    qs.reports.html(returns=daily_returns, output=report_path)
    with open(report_path) as html_report:
        wandb.log({"performance_report": wandb.Html(html_report)})

    metrics_to_log.update(
        {
            "sharpe_ratio": qs.stats.sharpe(daily_returns),
            "sortino_ratio": qs.stats.sortino(daily_returns),
            "cagr": qs.stats.cagr(daily_returns),
            "max_drawdown": qs.stats.max_drawdown(daily_returns),
            "calmar_ratio": qs.stats.calmar(daily_returns),
            "tail_ratio": qs.stats.tail_ratio(daily_returns),
            "common_sense_ratio": qs.stats.common_sense_ratio(daily_returns),
            "value_at_risk": qs.stats.value_at_risk(daily_returns),
            "conditional_value_at_risk": qs.stats.conditional_value_at_risk(
                daily_returns
            ),
            "information_ratio": qs.stats.information_ratio(
                daily_returns, benchmark=daily_returns
            ),
            "annual_volatility": qs.stats.volatility(daily_returns, annualize=True),
        }
    )

    # fig = plot_evolutions(df)
    # fig.show()

    # fig = plot_portfolio_composition(df)
    # fig.show()

    # fig = plot_actions(df)
    # fig.show()

    return metrics_to_log
