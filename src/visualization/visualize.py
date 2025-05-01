"""Contain visualization module."""

import altair as alt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


def load_data(
    path: str | None = None, data: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Load portfolio data for visualization.

    Args:
        path (str): Path to the CSV file containing the evaluation data.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing the DataFrame with portfolio data and a list of ticker symbols.
    """
    if data is None:
        assert path is not None, "Either df or path must be provided."
        data = pd.read_csv(path)

    # If date is a column, ensure it is the index
    if "date" in data.columns:
        data.set_index("date", inplace=True)

    # Extract tickers from the data
    tickers = [
        col.split("close_", 1)[1] for col in data.columns if col.startswith("close_")
    ]

    for ticker in tickers:
        prices = data[f"close_{ticker}"]
        returns = prices / prices.shift(1)
        shares = data[f"num_shares_owned_{ticker}"]
        orders = shares.diff()
        values = prices * shares

        data[f"daily_return_{ticker}"] = returns
        data[f"price_{ticker}"] = prices
        data[f"order_{ticker}"] = orders
        data[f"shares_{ticker}"] = shares
        data[f"value_{ticker}"] = values

    data["value_cash"] = data["cash"]
    data["portfolio_value"] = (
        sum(data[f"value_{ticker}"] for ticker in tickers) + data["value_cash"]
    )

    for ticker in tickers + ["cash"]:
        data[f"weight_{ticker}"] = data[f"value_{ticker}"] / data["portfolio_value"]

    for ticker in tickers:
        data[f"return_{ticker}"] = (
            data[f"price_{ticker}"] / data[f"price_{ticker}"].iloc[0]
        )
    data["portfolio_return"] = data["portfolio_value"] / data["portfolio_value"].iloc[0]

    return data, tickers


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


def visualize(
    data_path: str | None = None,
    report_path: str | None = None,
    data: pd.DataFrame | None = None,
) -> None:
    """Visualize the portfolio data.

    Args:
        data_path (str): Path to the CSV file containing the evaluation data.
        report_path (str): Path to the HTML report. Defaults to None.
        data (pd.DataFrame): DataFrame containing the evaluation data.
    """
    data, all_tickers = load_data(path=data_path, data=data)

    st.subheader("Portfolio Return and Asset Returns Over Time")
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

    # Display the HTML content
    if report_path:
        st.subheader("HTML Report")
        with open(report_path) as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True)


def main():
    """Main function to run the Streamlit app."""
    st.title("Portfolio Analysis")

    # Streamlit widgets for user inputs
    data_path = st.text_input("Enter the path to the CSV file", value="")
    report_path = st.text_input("Enter the path to the HTML report", value="")

    visualize(data_path, report_path)


if __name__ == "__main__":
    main()
