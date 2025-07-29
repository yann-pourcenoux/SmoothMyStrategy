"""Contain visualization module.

This module provides comprehensive visualization capabilities for portfolio trading analysis,
including:

- Portfolio and asset returns over time
- Portfolio distribution across assets
- Raw trading actions visualization and analysis
- Buy/sell signals with executed trades
- Comparison between intended actions and executed orders

The action visualization functionality allows users to:
1. View raw action values (intended trades) over time
2. Analyze action statistics (mean, std, min, max, counts)
3. Compare intended actions vs actual executed orders
4. Understand when actions were constrained by cash or position limits
"""

import altair as alt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from evaluation.metrics import compute_daily_returns, compute_portfolio_value


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
    tickers = [col.split("close_", 1)[1] for col in data.columns if col.startswith("close_")]

    for ticker in tickers:
        prices = data[f"close_{ticker}"]
        # Keep simple price returns for individual ticker visualization
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

    # Use metrics function for proper portfolio value calculation
    data["portfolio_value"] = compute_portfolio_value(data)

    for ticker in tickers + ["cash"]:
        data[f"weight_{ticker}"] = data[f"value_{ticker}"] / data["portfolio_value"]

    # Keep simple cumulative returns for individual ticker visualization
    for ticker in tickers:
        data[f"return_{ticker}"] = data[f"price_{ticker}"] / data[f"price_{ticker}"].iloc[0]

    # Use metrics function for proper portfolio daily returns, then compute cumulative
    portfolio_daily_returns = compute_daily_returns(data)
    data["portfolio_daily_returns"] = portfolio_daily_returns
    # Calculate cumulative returns from daily returns: (1 + r1) * (1 + r2) * ... - 1
    data["portfolio_return"] = (1 + portfolio_daily_returns).cumprod()

    return data, tickers


def display_returns(data: pd.DataFrame, tickers_to_show: list[str]) -> None:
    """Display the returns of the portfolio and selected tickers.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        tickers_to_show (list[str]): List of tickers to show returns for.
    """
    columns_to_display = ["portfolio_return"] + [f"return_{ticker}" for ticker in tickers_to_show]
    st.line_chart(data[columns_to_display])


def get_hex_colors_from_colormap(num_colors: int, colormap_name: str = "viridis") -> list[str]:
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
                color=alt.condition(alt.datum.order > 0, alt.value("green"), alt.value("red")),
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


def display_actions(data: pd.DataFrame, tickers_to_show: list[str]) -> None:
    """Display the raw actions (intended trades) for the selected tickers over time.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        tickers_to_show (list[str]): List of tickers to show actions for.
    """
    action_columns = [f"action_{ticker}" for ticker in tickers_to_show]

    # Check if action columns exist in the data
    available_action_columns = [col for col in action_columns if col in data.columns]

    if not available_action_columns:
        st.warning("No action data available in the dataset.")
        return

    # Create tabs for different views
    tab1, tab2 = st.tabs(["Action Values", "Action vs Executed Comparison"])

    with tab1:
        # Create the chart data
        chart_data = data[available_action_columns].copy()

        # Rename columns for better display
        chart_data.columns = [col.replace("action_", "") for col in chart_data.columns]

        st.line_chart(chart_data)

        # Add some statistics
        st.write("**Action Statistics:**")
        stats_data = {}
        for ticker in tickers_to_show:
            action_col = f"action_{ticker}"
            if action_col in data.columns:
                actions = data[action_col]
                stats_data[ticker] = {
                    "Mean": f"{actions.mean():.4f}",
                    "Std": f"{actions.std():.4f}",
                    "Min": f"{actions.min():.4f}",
                    "Max": f"{actions.max():.4f}",
                    "Buy Actions": f"{(actions > 0).sum()}",
                    "Sell Actions": f"{(actions < 0).sum()}",
                    "Hold Actions": f"{(actions == 0).sum()}",
                }

        if stats_data:
            stats_df = pd.DataFrame(stats_data).T
            st.dataframe(stats_df)

    with tab2:
        # Compare actions vs executed orders
        st.write("**Comparison: Intended Actions vs Executed Orders**")

        for ticker in tickers_to_show:
            action_col = f"action_{ticker}"
            order_col = f"order_{ticker}"

            if action_col in data.columns and order_col in data.columns:
                st.write(f"**{ticker}**")

                # Create comparison chart
                comparison_data = pd.DataFrame(
                    {
                        f"Intended ({ticker})": data[action_col],
                        f"Executed ({ticker})": data[order_col],
                    }
                )

                st.line_chart(comparison_data)

                # Show correlation and statistics
                correlation = data[action_col].corr(data[order_col])
                st.write(f"Correlation between intended and executed: {correlation:.4f}")

                # Show cases where actions were constrained
                constrained = data[action_col] != data[order_col]
                st.write(
                    f"Actions constrained: {constrained.sum()} out of {len(data)} days ({100 * constrained.mean():.1f}%)"
                )
            else:
                st.write(f"Missing data for {ticker}")
                if action_col not in data.columns:
                    st.write(f"- No action data ({action_col})")
                if order_col not in data.columns:
                    st.write(f"- No order data ({order_col})")


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

    st.subheader("Raw Actions Over Time")
    tickers_to_show_actions = st.multiselect(
        "Select tickers to show actions for:",
        all_tickers,
        default=all_tickers,
        key="tickers_to_show_actions",
    )
    display_actions(data, tickers_to_show_actions)

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
