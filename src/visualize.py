"""Contain visualization module."""

from typing import Optional

import altair as alt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_data(
    periods: int = 252,
    tickers: Optional[list[str]] = None,
    start_price: float = 50.0,
) -> pd.DataFrame:
    """Load data for the visualization.

    Args:
        periods (int): Number of periods to generate data for. Defaults to 252.
        tickers (Optional[list[str]]): List of tickers to generate data for. Defaults
            to ["AAPL", "GOOGL"].
        start_price (float): Starting price for the tickers. Defaults to 50.0.

    Returns:
        pd.DataFrame: DataFrame containing the generated data.
    """
    if tickers is None:
        tickers = ["AAPL", "GOOGL"]

    data = {"date": pd.date_range(start="2023-01-01", periods=periods, freq="D")}

    for ticker in tickers:
        returns = (1 + np.random.normal(0.001, 0.02, periods)).cumprod()
        prices = start_price * returns
        orders = np.random.choice([1.0, 0.0, -1.0], size=periods, p=[0.05, 0.9, 0.05])
        shares = np.random.randint(1, 100, periods)
        values = prices * shares

        data.update(
            {
                f"return_{ticker}": returns,
                f"price_{ticker}": prices,
                f"order_{ticker}": orders,
                f"shares_{ticker}": shares,
                f"value_{ticker}": values,
            }
        )

    data["value_cash"] = np.random.normal(10, 3, periods)
    data["portfolio_value"] = (
        sum(data[f"value_{ticker}"] for ticker in tickers) + data["value_cash"]
    )

    for ticker in tickers + ["cash"]:
        data[f"weight_{ticker}"] = data[f"value_{ticker}"] / data["portfolio_value"]

    data["portfolio_return"] = data["portfolio_value"] / data["portfolio_value"][0]

    data = pd.DataFrame(data)
    data = data.set_index("date")
    return data


def display_returns(data: pd.DataFrame, tickers_to_show: list[str]) -> None:
    """Display the returns of the portfolio and selected tickers.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        tickers_to_show (list[str]): List of tickers to show returns for.
    """
    columns_to_display = ["portfolio_return"] + [
        f"return_{ticker}" for ticker in tickers_to_show
    ]
    st.line_chart(data[columns_to_display])


def get_hex_colors_from_colormap(
    num_colors: int, colormap_name: str = "viridis"
) -> list[str]:
    """Get hex colors from a colormap.

    Args:
        num_colors (int): Number of colors to generate.
        colormap_name (str): Name of the colormap to use. Defaults to "viridis".

    Returns:
        list[str]: List of hex color codes.
    """
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i / num_colors) for i in range(num_colors)]
    return [mcolors.to_hex(c) for c in colors]


def display_buy_sell_signals(data: pd.DataFrame, tickers_to_show: list[str]) -> None:
    """Display buy and sell signals for the selected tickers.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        tickers_to_show (list[str]): List of tickers to show signals for.
    """
    colors = get_hex_colors_from_colormap(len(tickers_to_show))

    chart = None
    for ticker, color in zip(tickers_to_show, colors):
        data["ticker"] = ticker
        lines = (
            alt.Chart(data.reset_index())
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(f"return_{ticker}:Q", title=None),
                color=alt.Color(
                    "ticker:N",
                    scale=alt.Scale(domain=tickers_to_show, range=colors),
                    legend=alt.Legend(title="Tickers"),
                ),
            )
            .mark_line()
        )
        markers = (
            alt.Chart(data.reset_index())
            .transform_fold([f"order_{ticker}"], as_=["variable", "order"])
            .transform_filter(alt.datum.order != 0)
            .transform_calculate(price=f'datum["return_{ticker}"]')
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("price:Q", title=None),
                color=alt.condition(
                    alt.datum.order > 0, alt.value("green"), alt.value("red")
                ),
                shape=alt.condition(
                    alt.datum.order > 0,
                    alt.value("triangle-up"),
                    alt.value("triangle-down"),
                ),
            )
            .mark_point(size=100)
        )
        chart = lines + markers if chart is None else chart + lines + markers

    st.altair_chart(chart, use_container_width=True)


def plot_portfolio_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot the distribution of the portfolio in different assets over time.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        go.Figure: Plotly Figure object.
    """
    col_names = [col_name for col_name in df.columns if col_name.startswith("weight_")]
    fig = px.area(
        df,
        y=col_names,
        title="Portfolio Distribution Over Time",
        labels={"value": "Weight", "variable": "Asset"},
    )
    return fig


def main():
    """Main function to run the Streamlit app."""
    st.title("Portfolio Analysis")
    data = load_data()

    st.subheader("Portfolio Return and Asset Returns Over Time")
    all_tickers = ["AAPL", "GOOGL"]
    tickers_to_show_returns = st.multiselect(
        "Select tickers to show:",
        all_tickers,
        default=all_tickers,
        key="tickers_to_show_returns",
    )
    display_returns(data, tickers_to_show_returns)

    st.subheader("Portfolio Distribution Over Time")
    fig = plot_portfolio_distribution(data)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Asset Returns Over Time with Buy/Sell Signals")
    tickers_to_show_signals = st.multiselect(
        "Select tickers to show:",
        all_tickers,
        default=all_tickers,
        key="tickers_to_show_signals",
    )
    display_buy_sell_signals(data, tickers_to_show_signals)


if __name__ == "__main__":
    main()
