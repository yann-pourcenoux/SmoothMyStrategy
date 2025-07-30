import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.hierarchical_portfolio import HRPOpt

warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="Fund Analysis with HRP Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Risk-free rate for Sharpe ratio calculation
RISK_FREE_RATE = 0.025


# Fund tickers and names
@st.cache_data
def get_fund_data():
    """Return fund and benchmark data mappings."""
    tickers = {
        "0P0000X5RL.ST": "Avanza 75",
        "0P000151K0.F": "UB Infra A",
        "0P0001P9AO": "SEB Blockchain A",
        "0P0001H4TL.ST": "Avanza Emerging Markets",
        "6AQQ.DE": "Amundi Index Solutions - Amundi Nasdaq-100 ETF-C EUR",
        "A500.MI": "Amundi Index Solutions - Amundi S&P 500 UCITS ETF C EUR",
        "PRAE.DE": "Amundi Index Solutions - Amundi Prime Europe UCITS ETF DR (C)",
        "AMEM.DE": "Amundi Index Solutions - Amundi MSCI Emerging Markets UCITS ETF-C EUR",
        "CNAA.DE": "Amundi MSCI China A UCITS ETF Acc",
        "0P0000X5RM.ST": "Avanza 100",
        "0P0001BM0X.ST": "Avanza Auto 5",
        "0P0001BM0Y.ST": "Avanza Auto 6",
        "0P0001QD8L.ST": "Avanza Disruptive Innovation",
        "0P0001J6WY.ST": "Avanza Europa",
        "0P0001OD91.ST": "Avanza Fastigheter",
        "0P0001ECQR.ST": "Avanza Global",
        "0P0001QZW9.ST": "Avanza Healthcare",
        "0P0001QKMO.ST": "Avanza Impact",
        "0P0001L5HQ.ST": "Avanza Småbolag",
        "0P0001N85L.ST": "Avanza Sverige",
        "0P0001IVD1.ST": "Avanza USA",
        "0P0001K2MJ.ST": "Avanza World Tech",
        "0P00005U1J.ST": "Avanza Zero",
        "0P0001P9AP": "SEB Blockchain B",
    }

    benchmark_indices = {
        "^GSPC": "S&P 500",
        "^NDX": "NASDAQ 100",
        "^GDAXI": "DAX (Germany)",
        "SPY": "SPDR S&P 500 ETF",
    }

    return tickers, benchmark_indices


@st.cache_data
def download_fund_data(tickers, benchmarks, start_date, end_date, selected_funds=None):
    """Download fund and benchmark price data."""
    # Filter tickers if specific funds are selected
    if selected_funds:
        filtered_tickers = {k: v for k, v in tickers.items() if v in selected_funds}
    else:
        filtered_tickers = tickers

    # Download fund data
    ticker_list = list(filtered_tickers.keys())

    with st.spinner("Downloading fund data..."):
        try:
            fund_data = yf.download(
                ticker_list, start=start_date, end=end_date, auto_adjust=True, repair=True
            )

            if len(ticker_list) == 1:
                # Handle single ticker case
                fund_close = pd.DataFrame(fund_data["Close"])
                fund_close.columns = [filtered_tickers[ticker_list[0]]]
            else:
                fund_close = fund_data["Close"]
                fund_close.columns = [filtered_tickers[ticker] for ticker in fund_close.columns]
        except Exception as e:
            st.error(f"Error downloading fund data: {e}")
            return None, None

    # Download benchmark data
    benchmark_list = list(benchmarks.keys())

    with st.spinner("Downloading benchmark data..."):
        try:
            benchmark_data = yf.download(
                benchmark_list, start=start_date, end=end_date, auto_adjust=True, repair=True
            )

            if len(benchmark_list) == 1:
                benchmark_close = pd.DataFrame(benchmark_data["Close"])
                benchmark_close.columns = [benchmarks[benchmark_list[0]]]
            else:
                benchmark_close = benchmark_data["Close"]
                benchmark_close.columns = [benchmarks[ticker] for ticker in benchmark_close.columns]
        except Exception as e:
            st.error(f"Error downloading benchmark data: {e}")
            return fund_close, None

    return fund_close, benchmark_close


def calculate_daily_returns(close_prices):
    """Calculate daily returns from close prices."""
    daily_returns = close_prices.pct_change(fill_method=None)
    daily_returns = daily_returns.dropna(how="all")

    # Filter columns with sufficient data
    min_data_points = len(daily_returns) * 0.7
    valid_columns = daily_returns.count() >= min_data_points
    daily_returns = daily_returns.loc[:, valid_columns]

    return daily_returns


def plot_correlation_matrix(correlation_matrix):
    """Create interactive correlation matrix heatmap."""
    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix of Daily Returns",
    )
    fig.update_layout(height=600)
    return fig


def plot_cumulative_returns(fund_returns, benchmark_returns=None):
    """Create interactive cumulative returns plot."""
    cumulative_returns = (1 + fund_returns).cumprod() - 1

    fig = go.Figure()

    # Add fund returns
    for fund in cumulative_returns.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[fund] * 100,
                mode="lines",
                name=fund,
                line=dict(width=2),
            )
        )

    # Add benchmark returns if provided
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
        for benchmark in benchmark_cumulative.columns:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative[benchmark] * 100,
                    mode="lines",
                    name=f"{benchmark} (Benchmark)",
                    line=dict(width=3, dash="dash"),
                )
            )

    fig.update_layout(
        title="Cumulative Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=600,
        hovermode="x unified",
    )

    return fig


def plot_risk_return_scatter(total_returns, volatility, sharpe_ratio):
    """Create interactive risk-return scatter plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=volatility,
            y=total_returns,
            mode="markers+text",
            text=total_returns.index,
            textposition="top center",
            marker=dict(
                size=12,
                color=sharpe_ratio,
                colorscale="RdYlGn",
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=1, color="black"),
            ),
            hovertemplate="<b>%{text}</b><br>"
            + "Volatility: %{x:.2f}%<br>"
            + "Total Return: %{y:.2f}%<br>"
            + "Sharpe Ratio: %{marker.color:.3f}<extra></extra>",
        )
    )

    # Add risk-free rate line
    fig.add_hline(
        y=RISK_FREE_RATE * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Risk-Free Rate ({RISK_FREE_RATE * 100:.1f}%)",
    )

    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Total Return (%)",
        height=600,
    )

    return fig


def perform_eigenvalue_analysis(correlation_matrix):
    """Perform eigenvalue decomposition and create visualizations."""
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]

    explained_variance_ratio = eigenvalues_sorted / eigenvalues_sorted.sum()
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Scree Plot", "Explained Variance Ratio", "Cumulative Explained Variance"],
    )

    # Scree plot
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(eigenvalues_sorted) + 1)),
            y=eigenvalues_sorted,
            mode="lines+markers",
            name="Eigenvalues",
        ),
        row=1,
        col=1,
    )

    # Explained variance ratio
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(explained_variance_ratio) + 1)),
            y=explained_variance_ratio,
            name="Explained Variance",
        ),
        row=1,
        col=2,
    )

    # Cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance_ratio) + 1)),
            y=cumulative_variance_ratio,
            mode="lines+markers",
            name="Cumulative Variance",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(height=400, showlegend=False)

    return fig, eigenvalues_sorted, eigenvectors_sorted, explained_variance_ratio


def perform_hrp_optimization(daily_returns, correlation_matrix):
    """Perform HRP optimization and create visualizations."""
    hrp = HRPOpt(returns=daily_returns)
    weights = hrp.optimize()

    # Calculate performance
    expected_return, volatility, sharpe_ratio = hrp.portfolio_performance(
        risk_free_rate=RISK_FREE_RATE
    )

    # Create visualizations
    weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
    weights_df = weights_df.sort_values("Weight", ascending=False)

    # Portfolio weights bar chart
    fig_weights = go.Figure()
    fig_weights.add_trace(
        go.Bar(x=weights_df.index, y=weights_df["Weight"] * 100, name="Portfolio Weights")
    )
    fig_weights.update_layout(
        title="HRP Portfolio Weights",
        xaxis_title="Assets",
        yaxis_title="Weight (%)",
        xaxis_tickangle=-45,
        height=500,
    )

    # Portfolio allocation pie chart
    significant_weights = {k: v for k, v in weights.items() if v > 0.01}
    other_weight = sum(v for v in weights.values() if v <= 0.01)
    if other_weight > 0:
        significant_weights["Others"] = other_weight

    fig_pie = go.Figure()
    fig_pie.add_trace(
        go.Pie(
            labels=list(significant_weights.keys()),
            values=list(significant_weights.values()),
            hole=0.3,
        )
    )
    fig_pie.update_layout(title="HRP Portfolio Allocation", height=500)

    # Performance comparison
    n_assets = len(daily_returns.columns)
    equal_weights = np.ones(n_assets) / n_assets
    equal_weight_returns = (daily_returns * equal_weights).sum(axis=1)
    hrp_weights_array = np.array([weights[col] for col in daily_returns.columns])
    hrp_returns = (daily_returns * hrp_weights_array).sum(axis=1)

    eq_cum_returns = (1 + equal_weight_returns).cumprod()
    hrp_cum_returns = (1 + hrp_returns).cumprod()

    fig_comparison = go.Figure()
    fig_comparison.add_trace(
        go.Scatter(
            x=eq_cum_returns.index,
            y=eq_cum_returns,
            mode="lines",
            name="Equal Weight",
            line=dict(width=3),
        )
    )
    fig_comparison.add_trace(
        go.Scatter(
            x=hrp_cum_returns.index, y=hrp_cum_returns, mode="lines", name="HRP", line=dict(width=3)
        )
    )
    fig_comparison.update_layout(
        title="HRP vs Equal Weight Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=500,
    )

    return weights, expected_return, volatility, sharpe_ratio, fig_weights, fig_pie, fig_comparison


def perform_max_sharpe_optimization(daily_returns):
    """Perform Max Sharpe Ratio optimization using Efficient Frontier."""
    # Calculate expected returns and covariance matrix (annualized)
    # PyPortfolioOpt expects annualized returns by default
    mu = expected_returns.mean_historical_return(daily_returns, returns_data=True, frequency=252)
    S = risk_models.sample_cov(daily_returns, returns_data=True, frequency=252)

    # Add regularization to handle numerical issues
    # Add small values to diagonal to ensure positive definiteness
    regularization = 1e-5
    S = S + regularization * np.eye(S.shape[0])

    # Initialize Efficient Frontier with weight bounds to prevent extreme allocations
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))  # , solver="ECOS")

    # Optimize for maximal Sharpe ratio
    _ = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Calculate performance
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
        risk_free_rate=RISK_FREE_RATE, verbose=False
    )

    # Create visualizations
    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["Weight"])
    weights_df = weights_df[weights_df["Weight"] > 0]  # Only show non-zero weights
    weights_df = weights_df.sort_values("Weight", ascending=False)

    # Portfolio weights bar chart
    fig_weights = go.Figure()
    fig_weights.add_trace(
        go.Bar(
            x=weights_df.index,
            y=weights_df["Weight"] * 100,
            name="Portfolio Weights",
            marker_color="lightblue",
        )
    )
    fig_weights.update_layout(
        title="Max Sharpe Portfolio Weights",
        xaxis_title="Assets",
        yaxis_title="Weight (%)",
        xaxis_tickangle=-45,
        height=500,
    )

    # Portfolio allocation pie chart
    fig_pie = go.Figure()
    fig_pie.add_trace(go.Pie(labels=weights_df.index, values=weights_df["Weight"], hole=0.3))
    fig_pie.update_layout(title="Max Sharpe Portfolio Allocation", height=500)

    # Performance comparison with equal weight and individual assets
    n_assets = len(daily_returns.columns)
    equal_weights = np.ones(n_assets) / n_assets
    equal_weight_returns = (daily_returns * equal_weights).sum(axis=1)

    # Max Sharpe portfolio returns
    max_sharpe_weights_array = np.array(
        [cleaned_weights.get(col, 0) for col in daily_returns.columns]
    )
    max_sharpe_returns = (daily_returns * max_sharpe_weights_array).sum(axis=1)

    eq_cum_returns = (1 + equal_weight_returns).cumprod()
    max_sharpe_cum_returns = (1 + max_sharpe_returns).cumprod()

    fig_comparison = go.Figure()
    fig_comparison.add_trace(
        go.Scatter(
            x=eq_cum_returns.index,
            y=eq_cum_returns,
            mode="lines",
            name="Equal Weight",
            line=dict(width=3, color="blue"),
        )
    )
    fig_comparison.add_trace(
        go.Scatter(
            x=max_sharpe_cum_returns.index,
            y=max_sharpe_cum_returns,
            mode="lines",
            name="Max Sharpe",
            line=dict(width=3, color="red"),
        )
    )
    fig_comparison.update_layout(
        title="Max Sharpe vs Equal Weight Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=500,
    )

    # Create efficient frontier visualization
    fig_frontier = create_efficient_frontier_plot(mu, S, daily_returns.columns)

    return (
        cleaned_weights,
        expected_return,
        volatility,
        sharpe_ratio,
        fig_weights,
        fig_pie,
        fig_comparison,
        fig_frontier,
        mu,
        S,
    )


def create_efficient_frontier_plot(mu, S, asset_names):
    """Create efficient frontier visualization."""
    # Add regularization to ensure numerical stability
    regularization = 1e-5
    S_reg = S + regularization * np.eye(S.shape[0])

    # Calculate efficient frontier points
    ret_range = np.linspace(mu.min() * 1.1, mu.max() * 0.9, 50)  # Reduced points for stability
    frontier_volatility = []
    frontier_returns = []

    for target_return in ret_range:
        ef_temp = EfficientFrontier(mu, S_reg, weight_bounds=(0, 1))  # , solver="ECOS")
        ef_temp.efficient_return(target_return)
        _, vol, _ = ef_temp.portfolio_performance(verbose=False)
        frontier_volatility.append(vol)
        frontier_returns.append(target_return)

    # Max Sharpe point
    ef_max_sharpe = EfficientFrontier(mu, S_reg, weight_bounds=(0, 1))  # , solver="ECOS")
    ef_max_sharpe.max_sharpe()
    max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = ef_max_sharpe.portfolio_performance(
        verbose=False
    )
    # Min volatility point
    ef_min_vol = EfficientFrontier(mu, S_reg, weight_bounds=(0, 1))  # , solver="ECOS")
    ef_min_vol.min_volatility()
    min_vol_return, min_vol_vol, min_vol_ratio = ef_min_vol.portfolio_performance(verbose=False)

    # Individual assets
    individual_returns = mu.values
    individual_volatility = np.sqrt(np.diag(S.values))

    fig = go.Figure()

    # Plot efficient frontier
    fig.add_trace(
        go.Scatter(
            x=np.array(frontier_volatility) * 100,
            y=np.array(frontier_returns) * 100,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="blue", width=3),
        )
    )

    # Plot individual assets
    fig.add_trace(
        go.Scatter(
            x=individual_volatility * 100,
            y=individual_returns * 100,
            mode="markers+text",
            text=asset_names,
            textposition="top center",
            name="Individual Assets",
            marker=dict(size=8, color="lightgray", line=dict(width=1, color="black")),
        )
    )

    # Plot Max Sharpe point
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_vol * 100],
            y=[max_sharpe_return * 100],
            mode="markers",
            name=f"Max Sharpe (SR: {max_sharpe_ratio:.3f})",
            marker=dict(size=15, color="red", symbol="star"),
        )
    )

    # Plot Min Volatility point
    fig.add_trace(
        go.Scatter(
            x=[min_vol_vol * 100],
            y=[min_vol_return * 100],
            mode="markers",
            name=f"Min Volatility (SR: {min_vol_ratio:.3f})",
            marker=dict(size=15, color="green", symbol="diamond"),
        )
    )

    # Add risk-free rate line (Capital Allocation Line)
    max_vol = max(max(individual_volatility), max_sharpe_vol) * 100
    cal_x = np.linspace(0, max_vol, 100)
    cal_y = RISK_FREE_RATE * 100 + (max_sharpe_return - RISK_FREE_RATE) * 100 * (
        cal_x / (max_sharpe_vol * 100)
    )

    fig.add_trace(
        go.Scatter(
            x=cal_x,
            y=cal_y,
            mode="lines",
            name="Capital Allocation Line",
            line=dict(color="orange", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Efficient Frontier with Optimal Portfolios",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=600,
        hovermode="closest",
    )

    return fig


def main():
    """Main Streamlit application."""
    st.title("📈 Fund Analysis with Hierarchical Risk Parity")
    st.markdown("### Analyze fund performance and optimize portfolios using HRP")

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    # Get fund data
    tickers, benchmark_indices = get_fund_data()

    # Fund selection
    st.sidebar.subheader("Fund Selection")
    fund_names = list(tickers.values())
    selected_funds = st.sidebar.multiselect(
        "Select funds to analyze:",
        fund_names,
        default=fund_names,
        help="Choose which funds to include in the analysis",
    )

    if len(selected_funds) < 2:
        st.error("Please select at least 2 funds for analysis.")
        return

    # Date range selection
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", value=pd.to_datetime("2024-01-01"), help="Start date for data download"
    )
    end_date = st.sidebar.date_input(
        "End Date", value=pd.to_datetime("2025-07-30"), help="End date for data download"
    )

    # Risk-free rate
    st.sidebar.subheader("Risk Parameters")
    risk_free_rate = (
        st.sidebar.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Annual risk-free rate for Sharpe ratio calculation",
        )
        / 100
    )

    # Update global risk-free rate
    global RISK_FREE_RATE
    RISK_FREE_RATE = risk_free_rate

    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_correlation = st.sidebar.checkbox("Show Correlation Analysis", value=True)
    show_eigenvalue = st.sidebar.checkbox("Show Eigenvalue Analysis", value=True)
    show_hrp = st.sidebar.checkbox("Show HRP Optimization", value=True)
    show_max_sharpe = st.sidebar.checkbox("Show Max Sharpe Optimization", value=True)

    # Download data
    if st.sidebar.button("🔄 Load Data", type="primary"):
        fund_close, benchmark_close = download_fund_data(
            tickers, benchmark_indices, start_date, end_date, selected_funds
        )

        if fund_close is None:
            st.error("Failed to download fund data. Please check your selection.")
            return

        # Store data in session state
        st.session_state["fund_close"] = fund_close
        st.session_state["benchmark_close"] = benchmark_close
        st.session_state["data_loaded"] = True
        st.success(
            f"Data loaded successfully! {len(fund_close.columns)} funds, {len(fund_close)} trading days."
        )

    # Check if data is loaded
    if "data_loaded" not in st.session_state:
        st.info("👆 Please configure your analysis parameters and click 'Load Data' to begin.")
        return

    fund_close = st.session_state["fund_close"]
    benchmark_close = st.session_state["benchmark_close"]

    # Calculate returns
    fund_daily_returns = calculate_daily_returns(fund_close)
    benchmark_daily_returns = (
        calculate_daily_returns(benchmark_close) if benchmark_close is not None else None
    )

    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Performance Overview",
            "🔗 Correlation Analysis",
            "🧮 Eigenvalue Analysis",
            "⚖️ HRP Optimization",
            "🎯 Max Sharpe Optimization",
        ]
    )

    with tab1:
        st.header("Performance Overview")

        # Performance metrics
        cumulative_returns = (1 + fund_daily_returns).cumprod() - 1
        total_returns = cumulative_returns.iloc[-1] * 100
        volatility = fund_daily_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (fund_daily_returns.mean() * 252 - RISK_FREE_RATE) / (
            fund_daily_returns.std() * np.sqrt(252)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Performer", total_returns.idxmax(), f"{total_returns.max():.2f}%")

        with col2:
            st.metric("Lowest Volatility", volatility.idxmin(), f"{volatility.min():.2f}%")

        with col3:
            st.metric("Best Sharpe Ratio", sharpe_ratio.idxmax(), f"{sharpe_ratio.max():.3f}")

        # Cumulative returns plot
        st.subheader("Cumulative Returns")
        fig_returns = plot_cumulative_returns(fund_daily_returns, benchmark_daily_returns)
        st.plotly_chart(fig_returns, use_container_width=True)

        # Risk-return scatter
        st.subheader("Risk-Return Profile")
        fig_scatter = plot_risk_return_scatter(total_returns, volatility, sharpe_ratio)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Performance table
        st.subheader("Performance Summary")
        performance_df = pd.DataFrame(
            {
                "Fund": total_returns.index,
                "Total Return (%)": total_returns.values,
                "Volatility (%)": volatility.values,
                "Sharpe Ratio": sharpe_ratio.values,
            }
        ).sort_values("Total Return (%)", ascending=False)

        st.dataframe(performance_df, use_container_width=True)

    with tab2:
        if show_correlation:
            st.header("Correlation Analysis")

            correlation_matrix = fund_daily_returns.corr()

            # Correlation statistics
            col1, col2, col3 = st.columns(3)

            upper_tri = correlation_matrix.values[
                np.triu_indices_from(correlation_matrix.values, k=1)
            ]

            with col1:
                st.metric("Average Correlation", f"{upper_tri.mean():.3f}")

            with col2:
                st.metric("Max Correlation", f"{upper_tri.max():.3f}")

            with col3:
                st.metric("Min Correlation", f"{upper_tri.min():.3f}")

            # Correlation heatmap
            fig_corr = plot_correlation_matrix(correlation_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Store correlation matrix for other analyses
            st.session_state["correlation_matrix"] = correlation_matrix
        else:
            st.info("Correlation analysis is disabled. Enable it in the sidebar to view.")

    with tab3:
        if show_eigenvalue and "correlation_matrix" in st.session_state:
            st.header("Eigenvalue Analysis")

            correlation_matrix = st.session_state["correlation_matrix"]
            fig_eigen, eigenvalues, eigenvectors, explained_var = perform_eigenvalue_analysis(
                correlation_matrix
            )

            # Eigenvalue statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Largest Eigenvalue", f"{eigenvalues[0]:.3f}")

            with col2:
                st.metric("Smallest Eigenvalue", f"{eigenvalues[-1]:.3f}")

            with col3:
                st.metric("PC1 Variance Explained", f"{explained_var[0]:.1%}")

            with col4:
                st.metric("Top 3 PCs Cumulative", f"{explained_var[:3].sum():.1%}")

            # Eigenvalue plots
            st.plotly_chart(fig_eigen, use_container_width=True)

            # Principal component loadings
            st.subheader("Principal Component Loadings (First 3 Components)")
            loadings_df = pd.DataFrame(
                eigenvectors[:, :3],
                index=correlation_matrix.index,
                columns=[f"PC{i + 1}" for i in range(3)],
            )
            st.dataframe(loadings_df.round(3), use_container_width=True)

        else:
            st.info(
                "Eigenvalue analysis requires correlation matrix. Enable correlation analysis first."
            )

    with tab4:
        if show_hrp and "correlation_matrix" in st.session_state:
            st.header("Hierarchical Risk Parity Optimization")

            correlation_matrix = st.session_state["correlation_matrix"]

            with st.spinner("Optimizing portfolio using HRP..."):
                (
                    weights,
                    expected_return,
                    hrp_volatility,
                    hrp_sharpe,
                    fig_weights,
                    fig_pie,
                    fig_comparison,
                ) = perform_hrp_optimization(fund_daily_returns, correlation_matrix)

            # HRP performance metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Expected Annual Return", f"{expected_return * 100:.2f}%")

            with col2:
                st.metric("Annual Volatility", f"{hrp_volatility * 100:.2f}%")

            with col3:
                st.metric("Sharpe Ratio", f"{hrp_sharpe:.3f}")

            # Portfolio weights visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(fig_weights, use_container_width=True)

            with col2:
                st.plotly_chart(fig_pie, use_container_width=True)

            # Performance comparison
            st.subheader("HRP vs Equal Weight Performance")
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Portfolio weights table
            st.subheader("Optimal Portfolio Weights")
            weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
            weights_df["Weight (%)"] = weights_df["Weight"] * 100
            weights_df = weights_df.sort_values("Weight", ascending=False)
            st.dataframe(weights_df, use_container_width=True)

        else:
            st.info(
                "HRP optimization requires correlation matrix. Enable correlation analysis first."
            )

    with tab5:
        if show_max_sharpe:
            st.header("Max Sharpe Ratio Optimization")
            st.info(
                "📊 Using annualized expected returns (252 trading days) and covariance matrix with numerical regularization for stability."
            )

            with st.spinner("Optimizing portfolio for maximum Sharpe ratio..."):
                try:
                    (
                        max_sharpe_weights,
                        ms_expected_return,
                        ms_volatility,
                        ms_sharpe,
                        ms_fig_weights,
                        ms_fig_pie,
                        ms_fig_comparison,
                        ms_fig_frontier,
                        mu,
                        S,
                    ) = perform_max_sharpe_optimization(fund_daily_returns)

                    # Max Sharpe performance metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Expected Annual Return", f"{ms_expected_return * 100:.2f}%")

                    with col2:
                        st.metric("Annual Volatility", f"{ms_volatility * 100:.2f}%")

                    with col3:
                        st.metric("Sharpe Ratio", f"{ms_sharpe:.3f}")

                    # Efficient Frontier visualization
                    st.subheader("Efficient Frontier")
                    st.plotly_chart(ms_fig_frontier, use_container_width=True)

                    # Portfolio weights visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(ms_fig_weights, use_container_width=True)

                    with col2:
                        st.plotly_chart(ms_fig_pie, use_container_width=True)

                    # Performance comparison
                    st.subheader("Max Sharpe vs Equal Weight Performance")
                    st.plotly_chart(ms_fig_comparison, use_container_width=True)

                    # Portfolio weights table
                    st.subheader("Optimal Portfolio Weights")
                    ms_weights_df = pd.DataFrame.from_dict(
                        max_sharpe_weights, orient="index", columns=["Weight"]
                    )
                    ms_weights_df = ms_weights_df[
                        ms_weights_df["Weight"] > 0
                    ]  # Only show non-zero weights
                    ms_weights_df["Weight (%)"] = ms_weights_df["Weight"] * 100
                    ms_weights_df = ms_weights_df.sort_values("Weight", ascending=False)
                    st.dataframe(ms_weights_df, use_container_width=True)

                    # Expected returns and risk metrics
                    st.subheader("Asset Expected Returns and Risk Metrics")
                    metrics_df = pd.DataFrame(
                        {
                            "Asset": mu.index,
                            "Expected Annual Return (%)": mu.values * 100,
                            "Annual Volatility (%)": np.sqrt(np.diag(S.values)) * 100,
                            "Sharpe Ratio": (mu.values - RISK_FREE_RATE)
                            / np.sqrt(np.diag(S.values)),
                        }
                    ).sort_values("Sharpe Ratio", ascending=False)
                    st.dataframe(metrics_df, use_container_width=True)

                    # Comparison with HRP if available
                    if "correlation_matrix" in st.session_state:
                        st.subheader("Optimization Methods Comparison")

                        # Get HRP results for comparison
                        hrp = HRPOpt(returns=fund_daily_returns)
                        hrp_weights = hrp.optimize()
                        hrp_expected_return, hrp_volatility, hrp_sharpe = hrp.portfolio_performance(
                            risk_free_rate=RISK_FREE_RATE
                        )

                        # Equal weight performance
                        n_assets = len(fund_daily_returns.columns)
                        equal_weights = np.ones(n_assets) / n_assets
                        equal_weight_returns = (fund_daily_returns * equal_weights).sum(axis=1)
                        eq_annual_return = equal_weight_returns.mean() * 252
                        eq_annual_vol = equal_weight_returns.std() * np.sqrt(252)
                        eq_sharpe = (eq_annual_return - RISK_FREE_RATE) / eq_annual_vol

                        comparison_df = pd.DataFrame(
                            {
                                "Strategy": ["Equal Weight", "HRP", "Max Sharpe"],
                                "Expected Return (%)": [
                                    eq_annual_return * 100,
                                    hrp_expected_return * 100,
                                    ms_expected_return * 100,
                                ],
                                "Volatility (%)": [
                                    eq_annual_vol * 100,
                                    hrp_volatility * 100,
                                    ms_volatility * 100,
                                ],
                                "Sharpe Ratio": [eq_sharpe, hrp_sharpe, ms_sharpe],
                                "Number of Assets": [
                                    sum(1 for w in equal_weights if w > 0.01),
                                    sum(1 for w in hrp_weights.values() if w > 0.01),
                                    sum(1 for w in max_sharpe_weights.values() if w > 0.01),
                                ],
                            }
                        )

                        # Highlight best performers
                        def highlight_max(s):
                            if s.name == "Sharpe Ratio":
                                is_max = s == s.max()
                                return ["background-color: lightgreen" if v else "" for v in is_max]
                            elif s.name == "Expected Return (%)":
                                is_max = s == s.max()
                                return ["background-color: lightblue" if v else "" for v in is_max]
                            elif s.name == "Volatility (%)":
                                is_min = s == s.min()
                                return [
                                    "background-color: lightyellow" if v else "" for v in is_min
                                ]
                            return ["" for _ in s]

                        st.dataframe(
                            comparison_df.style.apply(highlight_max, axis=0),
                            use_container_width=True,
                        )

                        st.caption(
                            "🟢 Best Sharpe Ratio | 🔵 Highest Expected Return | 🟡 Lowest Volatility"
                        )

                except Exception as e:
                    st.error(f"Error in Max Sharpe optimization: {str(e)}")
                    st.info(
                        "This might be due to insufficient data or numerical issues. Try selecting different funds or date ranges."
                    )

        else:
            st.info("Max Sharpe optimization is disabled. Enable it in the sidebar to view.")


if __name__ == "__main__":
    main()
