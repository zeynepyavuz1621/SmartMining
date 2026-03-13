"""
Smart Mining — Streamlit Web UI
YAP 471 Computational Finance Term Project
Team MetalMinds
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

from data_module     import download_prices, compute_returns, get_data_summary, ASSETS
from signal_module   import generate_signals, compute_signal_strength
from portfolio_module import build_portfolio
from risk_module     import apply_stop_loss, compute_portfolio_drawdown
from backtest_module import run_backtest, equal_weight_benchmark, compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Mining — MetalMinds",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark industrial background */
.stApp {
    background: #0a0e14;
    color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}

/* Header */
h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.2rem !important;
    letter-spacing: 0.06em;
    color: #e6a817 !important;
    margin-bottom: 0 !important;
}

h2 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.8rem !important;
    color: #58a6ff !important;
    letter-spacing: 0.04em;
}

h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6a817 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    border-radius: 6px;
}

.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6a817 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e6a817, #f0c040) !important;
    color: #0a0e14 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.05em;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 2rem !important;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Selectbox / sliders */
.stSelectbox label, .stSlider label, .stDateInput label, .stMultiSelect label {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Info / success / warning boxes */
.stAlert {
    background: #161b22 !important;
    border-radius: 8px !important;
}

/* Divider */
hr {
    border-color: #21262d !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏️ SMART MINING")
    st.markdown("*Mean Reversion on AI Infrastructure Metals*")
    st.markdown("---")

    st.markdown("### 📅 Date Range")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start", value=date(2021, 1, 4),
                                   min_value=date(2015, 1, 1), max_value=date(2025, 1, 1))
    with col_b:
        end_date = st.date_input("End", value=date(2025, 12, 31),
                                 min_value=date(2016, 1, 1), max_value=date(2025, 12, 31))

    st.markdown("### 📦 Asset Selection")
    all_tickers = list(ASSETS.keys())
    selected_tickers = st.multiselect(
        "Choose assets (5–15):",
        options=all_tickers,
        default=all_tickers,
        format_func=lambda x: f"{x} — {ASSETS[x]}"
    )

    st.markdown("### 🎛️ Signal Parameters")
    lookback  = st.slider("Mean Reversion Lookback (days)", 5, 60, 20, step=5,
                          help="Rolling window to compute mean and std")
    threshold = st.slider("Z-Score Threshold", 0.5, 3.0, 1.0, step=0.25,
                          help="How many std-devs before a signal fires")

    st.markdown("### 📐 Portfolio Parameters")
    est_window    = st.slider("Estimation Window (days)", 20, 120, 60, step=10)
    rebal_freq    = st.slider("Rebalance Every (days)",   1,  21,  5, step=1)
    risk_aversion = st.slider("Risk Aversion (λ)",        0.1, 5.0, 1.0, step=0.1)

    st.markdown("### 🛡️ Risk Control")
    stop_loss_pct = st.slider("Stop-Loss Threshold (%)", 3, 30, 10, step=1) / 100
    cooldown_days = st.slider("Cooldown After Stop-Loss (days)", 1, 20, 5)

    st.markdown("### 🔬 Backtest")
    train_ratio = st.slider("Train / Test Split", 0.5, 0.9, 0.7, step=0.05,
                            help="Fraction of data used for training (in-sample)")

    st.markdown("---")
    run_btn = st.button("▶ RUN BACKTEST", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ⛏️ SMART MINING")
st.markdown("**YAP 471 · Team MetalMinds** — Mean Reversion Strategy on AI Infrastructure Commodities")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    # Landing state
    st.info("👈  Configure parameters in the sidebar and press **RUN BACKTEST** to start.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📡 Signal")
        st.markdown("Mean reversion via **rolling z-score**. When a metal's price spikes too far above its historical average, we sell. When it falls too low, we buy.")
    with c2:
        st.markdown("#### 📊 Portfolio")
        st.markdown("**Markowitz optimiser** converts signals into daily weights. No short selling. Rebalanced every N days using a rolling estimation window.")
    with c3:
        st.markdown("#### 🛡️ Risk")
        st.markdown("**Stop-loss rule**: if an asset drops >10% from its 20-day peak, it is excluded for a cooldown period before re-entry is allowed.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
if len(selected_tickers) < 2:
    st.error("Please select at least 2 assets.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Download data
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📡 Downloading price data from Yahoo Finance…"):
    prices = download_prices(selected_tickers, str(start_date), str(end_date))

if prices.empty or len(prices) < est_window + 30:
    st.error("Not enough data. Try a wider date range or different assets.")
    st.stop()

# Keep only successfully downloaded tickers
available = [t for t in selected_tickers if t in prices.columns]
prices = prices[available]
returns = compute_returns(prices)

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Computing signals…"):
    signals  = generate_signals(prices, lookback=lookback, threshold=threshold)
    zscores  = compute_signal_strength(prices, lookback=lookback)

with st.spinner("📐 Building portfolio…"):
    weights_raw = build_portfolio(returns, signals,
                                  estimation_window=est_window,
                                  rebalance_freq=rebal_freq,
                                  risk_aversion=risk_aversion)

with st.spinner("🛡️ Applying stop-loss…"):
    weights = apply_stop_loss(prices, weights_raw,
                              stop_loss_pct=stop_loss_pct,
                              cooldown_days=cooldown_days)

with st.spinner("📊 Running backtest…"):
    port_returns = run_backtest(prices, weights)
    bmark_returns = equal_weight_benchmark(prices)

    # Align
    common = port_returns.index.intersection(bmark_returns.index)
    port_returns  = port_returns.loc[common]
    bmark_returns = bmark_returns.loc[common]

    # Train/test split
    split = int(len(common) * train_ratio)
    test_idx = common[split:]

    port_metrics_full  = compute_metrics(port_returns)
    bmark_metrics_full = compute_metrics(bmark_returns)
    port_metrics_test  = compute_metrics(port_returns.loc[test_idx])
    bmark_metrics_test = compute_metrics(bmark_returns.loc[test_idx])

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance", "🎯 Signals", "⚖️ Weights", "🛡️ Risk", "📋 Data"
])

# ════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════════════
with tab1:
    st.markdown("## Performance Overview")

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return",     f"{port_metrics_full['Total Return (%)']:.1f}%",
                delta=f"{port_metrics_full['Total Return (%)'] - bmark_metrics_full['Total Return (%)']:.1f}% vs benchmark")
    col2.metric("Sharpe Ratio",     f"{port_metrics_full['Sharpe Ratio']:.3f}",
                delta=f"{port_metrics_full['Sharpe Ratio'] - bmark_metrics_full['Sharpe Ratio']:.3f} vs benchmark")
    col3.metric("Max Drawdown",     f"{port_metrics_full['Max Drawdown (%)']:.1f}%")
    col4.metric("Ann. Volatility",  f"{port_metrics_full['Ann. Volatility (%)']:.1f}%")

    st.markdown("---")

    # Cumulative return chart
    cum_port  = (1 + port_returns).cumprod()
    cum_bmark = (1 + bmark_returns).cumprod()
    split_date = common[split].strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_shape(
    type="line",
    x0=split_date, x1=split_date,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#444c56", width=1.5, dash="dash"),
)
    fig.add_annotation(
        x=split_date, y=1.0,
        xref="x", yref="paper",
        text="Train / Test Split",
        showarrow=False,
        font=dict(color="#8b949e", size=11),
        xanchor="left",
    )
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values,
                             name="Strategy", line=dict(color="#e6a817", width=2.5)))
    fig.add_trace(go.Scatter(x=cum_bmark.index, y=cum_bmark.values,
                             name="Equal-Weight Benchmark", line=dict(color="#58a6ff", width=1.5, dash="dot")))
    fig.update_layout(
        title="Cumulative Return (Normalised to 1)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown("### Full Period vs Out-of-Sample Metrics")
    metrics_df = pd.DataFrame({
        "Strategy (Full)":     port_metrics_full,
        "Benchmark (Full)":    bmark_metrics_full,
        "Strategy (Test)":     port_metrics_test,
        "Benchmark (Test)":    bmark_metrics_test,
    })
    st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=1, color="#1f3a1f")
                 .highlight_min(axis=1, color="#3a1f1f"), use_container_width=True)

    # Annual returns bar chart
    st.markdown("### Annual Returns")
    port_annual  = port_returns.resample("YE").apply(lambda r: (1+r).prod() - 1) * 100
    bmark_annual = bmark_returns.resample("YE").apply(lambda r: (1+r).prod() - 1) * 100
    years = [str(d.year) for d in port_annual.index]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Strategy",  x=years, y=port_annual.values,  marker_color="#e6a817"))
    fig2.add_trace(go.Bar(name="Benchmark", x=years, y=bmark_annual.values, marker_color="#58a6ff"))
    fig2.update_layout(
        barmode="group", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#161b22"), xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Return (%)"), height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 2 — SIGNALS
# ════════════════════════════════════════════════
with tab2:
    st.markdown("## Mean Reversion Signals")
    st.caption(f"Z-score lookback: **{lookback} days** · Threshold: **±{threshold}**")

    ticker_sel = st.selectbox("Select asset to inspect:", available,
                              format_func=lambda x: f"{x} — {ASSETS.get(x, x)}")

    if ticker_sel:
        z = zscores[ticker_sel].dropna()
        p = prices[ticker_sel]

        fig_sig = go.Figure()
        # Price
        fig_sig.add_trace(go.Scatter(x=p.index, y=p.values, name="Price",
                                     line=dict(color="#c9d1d9", width=1.5),
                                     yaxis="y2"))
        # Z-score
        fig_sig.add_trace(go.Scatter(x=z.index, y=z.values, name="Z-Score",
                                     line=dict(color="#e6a817", width=1.5)))
        # Threshold bands
        fig_sig.add_hline(y=threshold,  line_dash="dash", line_color="#f85149",
                          annotation_text="SELL zone", annotation_font_color="#f85149")
        fig_sig.add_hline(y=-threshold, line_dash="dash", line_color="#3fb950",
                          annotation_text="BUY zone", annotation_font_color="#3fb950")
        fig_sig.add_hline(y=0, line_color="#444c56", line_width=0.8)

        fig_sig.update_layout(
            title=f"{ticker_sel} — Z-Score & Price",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="IBM Plex Mono"),
            legend=dict(bgcolor="#161b22"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(title="Z-Score", gridcolor="#21262d"),
            yaxis2=dict(title="Price (USD)", overlaying="y", side="right", showgrid=False),
            height=420,
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # Signal distribution
        st.markdown("### Signal Distribution")
        sig_counts = signals[ticker_sel].value_counts().rename({1: "BUY", -1: "SELL", 0: "HOLD"})
        fig_pie = px.pie(values=sig_counts.values, names=sig_counts.index,
                         color=sig_counts.index,
                         color_discrete_map={"BUY": "#3fb950", "SELL": "#f85149", "HOLD": "#8b949e"})
        fig_pie.update_layout(paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Z-score heatmap (all assets, last 60 days)
    st.markdown("### Z-Score Heatmap — Last 60 Trading Days")
    z_recent = zscores.dropna().tail(60)
    fig_heat = go.Figure(go.Heatmap(
        z=z_recent.T.values,
        x=[str(d.date()) for d in z_recent.index],
        y=z_recent.columns.tolist(),
        colorscale=[[0, "#f85149"], [0.5, "#0d1117"], [1, "#3fb950"]],
        zmid=0, zmin=-3, zmax=3,
        colorbar=dict(title="Z-Score", tickfont=dict(color="#8b949e")),
    ))
    fig_heat.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        xaxis=dict(showticklabels=False), height=300,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 3 — WEIGHTS
# ════════════════════════════════════════════════
with tab3:
    st.markdown("## Portfolio Weights Over Time")

    # Stacked area chart
    fig_w = go.Figure()
    palette = ["#e6a817","#58a6ff","#3fb950","#f85149","#d2a8ff",
               "#ffa657","#79c0ff","#56d364","#ff7b72","#bc8cff"]
    for i, t in enumerate(available):
        fig_w.add_trace(go.Scatter(
            x=weights.index, y=weights[t].values,
            name=f"{t} — {ASSETS.get(t,t)}",
            stackgroup="one",
            line=dict(width=0.5, color=palette[i % len(palette)]),
            fillcolor=palette[i % len(palette)],
        ))
    fig_w.update_layout(
        title="Dynamic Portfolio Weights (Stacked)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", tickformat=".0%", range=[0, 1]),
        height=420,
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # Latest weights bar chart
    st.markdown("### Latest Portfolio Weights")
    latest_w = weights.iloc[-1].sort_values(ascending=False)
    fig_bar = go.Figure(go.Bar(
        x=latest_w.index,
        y=latest_w.values * 100,
        marker_color=[palette[i % len(palette)] for i in range(len(latest_w))],
        text=[f"{v*100:.1f}%" for v in latest_w.values],
        textposition="outside",
        textfont=dict(color="#c9d1d9"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        yaxis=dict(gridcolor="#21262d", title="Weight (%)"),
        xaxis=dict(gridcolor="#21262d"),
        height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 — RISK
# ════════════════════════════════════════════════
with tab4:
    st.markdown("## Risk Analysis")

    cum_port_r  = (1 + port_returns).cumprod()
    cum_bmark_r = (1 + bmark_returns).cumprod()
    dd_port  = compute_portfolio_drawdown(cum_port_r)  * 100
    dd_bmark = compute_portfolio_drawdown(cum_bmark_r) * 100

    # Drawdown chart
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_port.index,  y=dd_port.values,
                                name="Strategy",  line=dict(color="#e6a817", width=2),
                                fill="tozeroy", fillcolor="rgba(230,168,23,0.15)"))
    fig_dd.add_trace(go.Scatter(x=dd_bmark.index, y=dd_bmark.values,
                                name="Benchmark", line=dict(color="#58a6ff", width=1.5, dash="dot")))
    fig_dd.update_layout(
        title="Drawdown (%)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#161b22"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Drawdown (%)"),
        height=380,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Rolling Sharpe
    st.markdown("### Rolling 60-Day Sharpe Ratio")
    roll_sharpe = port_returns.rolling(60).apply(
        lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)
    fig_rs = go.Figure(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                                  line=dict(color="#e6a817", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(230,168,23,0.1)"))
    fig_rs.add_hline(y=0, line_color="#444c56")
    fig_rs.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Sharpe Ratio"),
        height=320,
    )
    st.plotly_chart(fig_rs, use_container_width=True)

    # Rolling volatility
    st.markdown("### Rolling 30-Day Annualised Volatility")
    roll_vol_port  = port_returns.rolling(30).std()  * np.sqrt(252) * 100
    roll_vol_bmark = bmark_returns.rolling(30).std() * np.sqrt(252) * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=roll_vol_port.index,  y=roll_vol_port.values,
                                 name="Strategy",  line=dict(color="#e6a817", width=2)))
    fig_vol.add_trace(go.Scatter(x=roll_vol_bmark.index, y=roll_vol_bmark.values,
                                 name="Benchmark", line=dict(color="#58a6ff", width=1.5, dash="dot")))
    fig_vol.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#161b22"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Volatility (%)"),
        height=320,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 5 — DATA
# ════════════════════════════════════════════════
with tab5:
    st.markdown("## Asset Data Summary")
    summary = get_data_summary(prices)
    summary_df = pd.DataFrame(summary).T
    st.dataframe(summary_df.style.format({
        "start_price":  "{:.2f}",
        "end_price":    "{:.2f}",
        "total_return": "{:.1f}%",
        "ann_vol":      "{:.1f}%",
        "n_obs":        "{:.0f}",
    }), use_container_width=True)

    st.markdown("### Raw Price Chart")
    fig_price = go.Figure()
    for i, t in enumerate(available):
        norm = prices[t] / prices[t].iloc[0]  # normalise to 1
        fig_price.add_trace(go.Scatter(x=norm.index, y=norm.values,
                                       name=t, line=dict(color=palette[i % len(palette)], width=1.5)))
    fig_price.update_layout(
        title="Normalised Price (Start = 1)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        height=420,
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Correlation heatmap
    st.markdown("### Return Correlation Matrix")
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(tickfont=dict(color="#8b949e")),
    ))
    fig_corr.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="IBM Plex Mono"),
        height=400,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Download buttons
    st.markdown("### Export Data")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        csv_prices = prices.to_csv().encode("utf-8")
        st.download_button("⬇ Download Prices CSV",  csv_prices,  "prices.csv",  "text/csv")
    with col_d2:
        csv_weights = weights.to_csv().encode("utf-8")
        st.download_button("⬇ Download Weights CSV", csv_weights, "weights.csv", "text/csv")
