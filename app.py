"""
Smart Mining — Streamlit Web UI
YAP 471 Computational Finance Term Project
Team MetalMinds
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import zipfile, io
from datetime import date

from data_module      import download_prices, compute_returns, get_data_summary, ASSETS
from signal_module    import generate_signals, compute_signal_strength, Z_THRESHOLD
from portfolio_module import build_portfolio, CASH
from risk_module      import apply_stop_loss, compute_portfolio_drawdown
from backtest_module  import run_backtest, equal_weight_benchmark, compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Default parameter values
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "start_date":     date(2021, 1, 4),
    "end_date":       date(2025, 12, 31),
    "lookback":       20,
    "est_window":     40,
    "rebal_freq":     5,
    "risk_aversion":  1.0,
    "stop_loss_pct":  10,
    "cooldown_days":  5,
}

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

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0a0e14; color: #c9d1d9; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #21262d; }

h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 3.2rem !important; letter-spacing: 0.06em; color: #e6a817 !important; margin-bottom: 0 !important; }
h2 { font-family: 'Bebas Neue', sans-serif !important; font-size: 1.8rem !important; color: #58a6ff !important; letter-spacing: 0.04em; }
h3 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; color: #8b949e !important; font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

[data-testid="metric-container"] { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; padding: 16px !important; }
[data-testid="metric-container"] label { color: #8b949e !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6a817 !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important; }

.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.05em; border-radius: 6px; }
.stTabs [aria-selected="true"] { background: #21262d !important; color: #e6a817 !important; }

.stButton > button {
    background: linear-gradient(135deg, #e6a817, #f0c040) !important;
    color: #0a0e14 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.05em;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 2rem !important;
}
div[data-testid="stButton"]:has(button[kind="secondary"]) button {
    background: #21262d !important;
    color: #8b949e !important;
    font-size: 0.85rem !important;
    padding: 0.3rem 1rem !important;
    border: 1px solid #30363d !important;
}
.stSelectbox label, .stSlider label, .stDateInput label, .stMultiSelect label {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared plot base
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", family="IBM Plex Mono"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    xaxis=dict(gridcolor="#21262d"),
)
PALETTE = [
    "#e6a817", "#58a6ff", "#3fb950", "#f85149", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff7b72", "#bc8cff",
]
CASH_COLOR = "#444c56"

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # checkbox defaults
    for k, v in [("dl_metrics", True), ("dl_cumret", True), ("dl_weights", True),
                 ("dl_signals", False), ("dl_zscores", False)]:
        if k not in st.session_state:
            st.session_state[k] = v
    # backtest results
    if "results" not in st.session_state:
        st.session_state["results"] = None

def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["results"] = None  # clear cached results on reset

init_defaults()

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
        start_date = st.date_input("Start", key="start_date",
                                   min_value=date(2015, 1, 1), max_value=date(2025, 1, 1))
    with col_b:
        end_date = st.date_input("End", key="end_date",
                                 min_value=date(2016, 1, 1), max_value=date(2025, 12, 31))

    st.markdown("### 📦 Asset Selection")
    selected_tickers = st.multiselect(
        "Choose assets:",
        options=list(ASSETS.keys()),
        default=list(ASSETS.keys()),
        format_func=lambda x: f"{x} — {ASSETS[x]}"
    )

    st.markdown("### 🎛️ Signal Parameters")
    lookback = st.slider("Mean Reversion Lookback (days)", 5, 40, key="lookback", step=5)
    st.caption(f"Z-score threshold fixed at **±{Z_THRESHOLD}** (90% confidence interval)")

    st.markdown("### 📐 Portfolio Parameters")
    est_window    = st.slider("Estimation Window (days)", 20, 60, key="est_window",    step=10)
    rebal_freq    = st.slider("Rebalance Every (days)",    1,  21, key="rebal_freq",    step=1)
    risk_aversion = st.slider("Risk Aversion (λ)",       0.1, 5.0, key="risk_aversion", step=0.1)

    st.markdown("### 🛡️ Risk Control")
    stop_loss_pct = st.slider("Stop-Loss Threshold (%)", 3, 30, key="stop_loss_pct", step=1)
    cooldown_days = st.slider("Cooldown After Stop-Loss (days)", 1, 20, key="cooldown_days")

    st.markdown("---")
    st.button("↺ Reset to Defaults", on_click=reset_defaults, use_container_width=True, type="secondary")
    run_btn = st.button("▶ RUN BACKTEST", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ⛏️ SMART MINING")
st.markdown("**YAP 471 · Team MetalMinds** — Mean Reversion Strategy on AI Infrastructure Commodities")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Landing
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn and st.session_state["results"] is None:
    st.info("👈  Configure parameters in the sidebar and press **RUN BACKTEST** to start.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📡 Signal")
        st.markdown(f"Mean reversion via **rolling z-score**. Threshold fixed at **±{Z_THRESHOLD}** (90% CI). When price is statistically too high, we sell. When too low, we buy.")
    with c2:
        st.markdown("#### 📊 Portfolio")
        st.markdown("**Markowitz optimiser** with **Cash** as an explicit asset. If nothing looks attractive, the strategy moves to cash.")
    with c3:
        st.markdown("#### 🛡️ Risk")
        st.markdown("**Stop-loss**: if an asset drops beyond the threshold from its 20-day peak, its weight flows to other assets or cash.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Run backtest only when button is pressed
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if len(selected_tickers) < 1:
        st.error("Please select at least 1 asset.")
        st.stop()
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    _lookback      = st.session_state["lookback"]
    _est_window    = st.session_state["est_window"]
    _rebal_freq    = st.session_state["rebal_freq"]
    _risk_aversion = st.session_state["risk_aversion"]
    _stop_loss_pct = st.session_state["stop_loss_pct"] / 100
    _cooldown_days = st.session_state["cooldown_days"]

    with st.spinner("📡 Downloading price data from Yahoo Finance…"):
        prices = download_prices(selected_tickers, str(start_date), str(end_date))

    if prices.empty or len(prices) < _est_window + 30:
        st.error("Not enough data. Try a wider date range or different assets.")
        st.stop()

    available = [t for t in selected_tickers if t in prices.columns]
    prices    = prices[available]
    returns   = compute_returns(prices)

    with st.spinner("⚙️ Computing signals…"):
        signals = generate_signals(prices, lookback=_lookback)
        zscores = compute_signal_strength(prices, lookback=_lookback)

    with st.spinner("📐 Building portfolio (with Cash option)…"):
        weights_raw = build_portfolio(returns, signals,
                                      estimation_window=_est_window,
                                      rebalance_freq=_rebal_freq,
                                      risk_aversion=_risk_aversion)

    with st.spinner("🛡️ Applying stop-loss…"):
        weights = apply_stop_loss(prices, weights_raw,
                                  stop_loss_pct=_stop_loss_pct,
                                  cooldown_days=_cooldown_days)

    with st.spinner("📊 Running backtest…"):
        prices_for_bt       = prices.copy()
        prices_for_bt[CASH] = 1.0
        port_returns        = run_backtest(prices_for_bt, weights)
        bmark_returns       = equal_weight_benchmark(prices)
        common              = port_returns.index.intersection(bmark_returns.index)
        port_returns        = port_returns.loc[common]
        bmark_returns       = bmark_returns.loc[common]
        port_metrics        = compute_metrics(port_returns)
        bmark_metrics       = compute_metrics(bmark_returns)

    # ── Cache all results in session_state ──────────────────────────────────
    st.session_state["results"] = {
        "prices":        prices,
        "returns":       returns,
        "signals":       signals,
        "zscores":       zscores,
        "weights":       weights,
        "port_returns":  port_returns,
        "bmark_returns": bmark_returns,
        "port_metrics":  port_metrics,
        "bmark_metrics": bmark_metrics,
        "available":     available,
        # save params for export readme
        "params": {
            "start_date":    str(start_date),
            "end_date":      str(end_date),
            "assets":        ", ".join(available),
            "lookback":      _lookback,
            "z_threshold":   Z_THRESHOLD,
            "est_window":    _est_window,
            "rebal_freq":    _rebal_freq,
            "risk_aversion": _risk_aversion,
            "stop_loss_pct": int(_stop_loss_pct * 100),
            "cooldown_days": _cooldown_days,
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# Load from cache
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["results"] is None:
    st.stop()

R             = st.session_state["results"]
prices        = R["prices"]
returns       = R["returns"]
signals       = R["signals"]
zscores       = R["zscores"]
weights       = R["weights"]
port_returns  = R["port_returns"]
bmark_returns = R["bmark_returns"]
port_metrics  = R["port_metrics"]
bmark_metrics = R["bmark_metrics"]
available     = R["available"]
params        = R["params"]

metal_cols  = [c for c in weights.columns if c != CASH]
cash_weight = weights[CASH] if CASH in weights.columns else pd.Series(0.0, index=weights.index)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance", "🎯 Signals", "⚖️ Weights", "🛡️ Risk", "📋 Data"
])

# ══════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════════
with tab1:
    st.markdown("## Performance Overview")

    avg_cash = float(cash_weight.mean() * 100)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return",
              f"{port_metrics['Total Return (%)']:.1f}%",
              delta=f"{port_metrics['Total Return (%)'] - bmark_metrics['Total Return (%)']:.1f}% vs benchmark")
    c2.metric("Sharpe Ratio",
              f"{port_metrics['Sharpe Ratio']:.3f}",
              delta=f"{port_metrics['Sharpe Ratio'] - bmark_metrics['Sharpe Ratio']:.3f} vs benchmark")
    c3.metric("Max Drawdown",    f"{port_metrics['Max Drawdown (%)']:.1f}%")
    c4.metric("Ann. Volatility", f"{port_metrics['Ann. Volatility (%)']:.1f}%")
    c5.metric("Avg Cash %",      f"{avg_cash:.1f}%")

    st.markdown("---")

    cum_port  = (1 + port_returns).cumprod()
    cum_bmark = (1 + bmark_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index,  y=cum_port.values,  name="Strategy",
                             line=dict(color="#e6a817", width=2.5)))
    fig.add_trace(go.Scatter(x=cum_bmark.index, y=cum_bmark.values, name="Equal-Weight Benchmark",
                             line=dict(color="#58a6ff", width=1.5, dash="dot")))
    fig.update_layout(**DARK_BG, title="Cumulative Return (Normalised to 1)", height=420)
    fig.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Performance Metrics")
    metrics_df = pd.DataFrame({"Strategy": port_metrics, "Benchmark": bmark_metrics})
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("### Annual Returns")
    port_annual  = port_returns.resample("YE").apply(lambda r: (1+r).prod() - 1) * 100
    bmark_annual = bmark_returns.resample("YE").apply(lambda r: (1+r).prod() - 1) * 100
    years = [str(d.year) for d in port_annual.index]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Strategy",  x=years, y=port_annual.values,  marker_color="#e6a817"))
    fig2.add_trace(go.Bar(name="Benchmark", x=years, y=bmark_annual.values, marker_color="#58a6ff"))
    fig2.update_layout(**DARK_BG, barmode="group", height=320)
    fig2.update_yaxes(gridcolor="#21262d", title_text="Return (%)")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════
# TAB 2 — SIGNALS
# ══════════════════════════════════════════════════
with tab2:
    st.markdown("## Mean Reversion Signals")
    st.caption(f"Lookback: **{params['lookback']} days** · Z-score threshold: **±{Z_THRESHOLD}** (90% CI)")

    ticker_sel = st.selectbox("Select asset to inspect:", available,
                              format_func=lambda x: f"{x} — {ASSETS.get(x, x)}")

    if ticker_sel:
        z = zscores[ticker_sel].dropna()
        p = prices[ticker_sel]

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=p.index, y=p.values, name="Price",
                                     line=dict(color="#c9d1d9", width=1.5), yaxis="y2"))
        fig_sig.add_trace(go.Scatter(x=z.index, y=z.values, name="Z-Score",
                                     line=dict(color="#e6a817", width=1.5)))
        fig_sig.add_hline(y= Z_THRESHOLD, line_dash="dash", line_color="#f85149",
                          annotation_text="SELL zone", annotation_font_color="#f85149")
        fig_sig.add_hline(y=-Z_THRESHOLD, line_dash="dash", line_color="#3fb950",
                          annotation_text="BUY zone",  annotation_font_color="#3fb950")
        fig_sig.add_hline(y=0, line_color="#444c56", line_width=0.8)
        fig_sig.update_layout(
            **DARK_BG,
            title=f"{ticker_sel} — Z-Score & Price",
            height=420,
            yaxis=dict(title="Z-Score", gridcolor="#21262d"),
            yaxis2=dict(title="Price (USD)", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        st.markdown("### Signal Distribution")
        sig_counts = signals[ticker_sel].value_counts().rename({1: "BUY", -1: "SELL", 0: "HOLD"})
        fig_pie = px.pie(values=sig_counts.values, names=sig_counts.index,
                         color=sig_counts.index,
                         color_discrete_map={"BUY": "#3fb950", "SELL": "#f85149", "HOLD": "#8b949e"})
        fig_pie.update_layout(paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

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
    fig_heat.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           font=dict(color="#8b949e", family="IBM Plex Mono"),
                           xaxis=dict(showticklabels=False), height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════
# TAB 3 — WEIGHTS
# ══════════════════════════════════════════════════
with tab3:
    st.markdown("## Portfolio Weights Over Time")
    st.caption("Grey area = Cash (strategy is not forced to stay invested)")

    fig_w = go.Figure()
    for i, t in enumerate(metal_cols):
        if t in weights.columns:
            fig_w.add_trace(go.Scatter(
                x=weights.index, y=weights[t].values,
                name=f"{t} — {ASSETS.get(t, t)}",
                stackgroup="one",
                line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]),
                fillcolor=PALETTE[i % len(PALETTE)],
            ))
    fig_w.add_trace(go.Scatter(
        x=weights.index, y=cash_weight.values,
        name="💵 Cash",
        stackgroup="one",
        line=dict(width=0.5, color=CASH_COLOR),
        fillcolor=CASH_COLOR,
    ))
    fig_w.update_layout(**DARK_BG, title="Dynamic Portfolio Weights incl. Cash", height=420)
    fig_w.update_yaxes(gridcolor="#21262d", tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("### Latest Portfolio Weights")
    latest_w = weights.iloc[-1].sort_values(ascending=False)
    colors = []
    for t in latest_w.index:
        if t == CASH:
            colors.append(CASH_COLOR)
        else:
            idx = metal_cols.index(t) if t in metal_cols else 0
            colors.append(PALETTE[idx % len(PALETTE)])
    labels = [("💵 Cash" if t == CASH else t) for t in latest_w.index]

    fig_bar = go.Figure(go.Bar(
        x=labels, y=latest_w.values * 100,
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in latest_w.values],
        textposition="outside",
        textfont=dict(color="#c9d1d9"),
    ))
    fig_bar.update_layout(**DARK_BG, height=320)
    fig_bar.update_yaxes(gridcolor="#21262d", title_text="Weight (%)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Cash Allocation Over Time")
    fig_cash = go.Figure(go.Scatter(
        x=cash_weight.index, y=cash_weight.values * 100,
        line=dict(color=CASH_COLOR, width=1.5),
        fill="tozeroy", fillcolor="rgba(68,76,86,0.3)",
    ))
    fig_cash.update_layout(**DARK_BG, title="Daily Cash Allocation (%)", height=280)
    fig_cash.update_yaxes(gridcolor="#21262d", title_text="Cash (%)", range=[0, 100])
    st.plotly_chart(fig_cash, use_container_width=True)

# ══════════════════════════════════════════════════
# TAB 4 — RISK
# ══════════════════════════════════════════════════
with tab4:
    st.markdown("## Risk Analysis")

    cum_port_r  = (1 + port_returns).cumprod()
    cum_bmark_r = (1 + bmark_returns).cumprod()
    dd_port     = compute_portfolio_drawdown(cum_port_r)  * 100
    dd_bmark    = compute_portfolio_drawdown(cum_bmark_r) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_port.index,  y=dd_port.values,  name="Strategy",
                                line=dict(color="#e6a817", width=2),
                                fill="tozeroy", fillcolor="rgba(230,168,23,0.15)"))
    fig_dd.add_trace(go.Scatter(x=dd_bmark.index, y=dd_bmark.values, name="Benchmark",
                                line=dict(color="#58a6ff", width=1.5, dash="dot")))
    fig_dd.update_layout(**DARK_BG, title="Drawdown (%)", height=380)
    fig_dd.update_yaxes(gridcolor="#21262d", title_text="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("### Rolling 60-Day Sharpe Ratio")
    roll_sharpe = port_returns.rolling(60).apply(
        lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)
    fig_rs = go.Figure(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                                  line=dict(color="#e6a817", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(230,168,23,0.1)"))
    fig_rs.add_hline(y=0, line_color="#444c56")
    fig_rs.update_layout(**DARK_BG, height=320)
    fig_rs.update_yaxes(gridcolor="#21262d", title_text="Sharpe Ratio")
    st.plotly_chart(fig_rs, use_container_width=True)

    st.markdown("### Rolling 30-Day Annualised Volatility")
    roll_vol_port  = port_returns.rolling(30).std()  * np.sqrt(252) * 100
    roll_vol_bmark = bmark_returns.rolling(30).std() * np.sqrt(252) * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=roll_vol_port.index,  y=roll_vol_port.values,
                                 name="Strategy",  line=dict(color="#e6a817", width=2)))
    fig_vol.add_trace(go.Scatter(x=roll_vol_bmark.index, y=roll_vol_bmark.values,
                                 name="Benchmark", line=dict(color="#58a6ff", width=1.5, dash="dot")))
    fig_vol.update_layout(**DARK_BG, height=320)
    fig_vol.update_yaxes(gridcolor="#21262d", title_text="Volatility (%)")
    st.plotly_chart(fig_vol, use_container_width=True)

# ══════════════════════════════════════════════════
# TAB 5 — DATA
# ══════════════════════════════════════════════════
with tab5:
    st.markdown("## Asset Data Summary")
    summary_df = pd.DataFrame(get_data_summary(prices)).T
    st.dataframe(summary_df.style.format({
        "start_price":  "{:.2f}",
        "end_price":    "{:.2f}",
        "total_return": "{:.1f}%",
        "ann_vol":      "{:.1f}%",
        "n_obs":        "{:.0f}",
    }), use_container_width=True)

    st.markdown("### Normalised Price Chart")
    fig_price = go.Figure()
    for i, t in enumerate(available):
        norm = prices[t] / prices[t].iloc[0]
        fig_price.add_trace(go.Scatter(x=norm.index, y=norm.values, name=t,
                                       line=dict(color=PALETTE[i % len(PALETTE)], width=1.5)))
    fig_price.update_layout(**DARK_BG, title="Normalised Price (Start = 1)", height=420)
    fig_price.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("### Return Correlation Matrix")
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(tickfont=dict(color="#8b949e")),
    ))
    fig_corr.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           font=dict(color="#c9d1d9", family="IBM Plex Mono"), height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Export ──────────────────────────────────────────────────────────────
    st.markdown("### Export Data")
    st.caption("Select files to include in the ZIP, then download.")

    col_e1, col_e2, col_e3, col_e4, col_e5 = st.columns(5)
    with col_e1:
        st.checkbox("📊 Metrics",            key="dl_metrics")
    with col_e2:
        st.checkbox("📈 Cumulative Returns",  key="dl_cumret")
    with col_e3:
        st.checkbox("⚖️ Weights",            key="dl_weights")
    with col_e4:
        st.checkbox("🎯 Signals",            key="dl_signals")
    with col_e5:
        st.checkbox("🔢 Z-Scores",           key="dl_zscores")

    # Build ZIP in memory (always ready, no button needed to re-run backtest)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        if st.session_state["dl_metrics"]:
            zf.writestr("metrics.csv",
                        pd.DataFrame({"Strategy": port_metrics,
                                      "Benchmark": bmark_metrics}).to_csv())

        if st.session_state["dl_cumret"]:
            zf.writestr("cumulative_returns.csv",
                        pd.DataFrame({
                            "Strategy":  (1 + port_returns).cumprod(),
                            "Benchmark": (1 + bmark_returns).cumprod(),
                        }).to_csv())

        if st.session_state["dl_weights"]:
            zf.writestr("weights.csv", weights.to_csv())

        if st.session_state["dl_signals"]:
            zf.writestr("signals.csv", signals.to_csv())

        if st.session_state["dl_zscores"]:
            zf.writestr("zscores.csv", zscores.to_csv())

        # always include prices + params
        zf.writestr("prices.csv", prices.to_csv())
        param_lines = [
            "Smart Mining — Backtest Export",
            "================================",
            f"Start Date:        {params['start_date']}",
            f"End Date:          {params['end_date']}",
            f"Assets:            {params['assets']}",
            f"Lookback:          {params['lookback']} days",
            f"Z-Score Threshold: {params['z_threshold']} (90% CI, fixed)",
            f"Estimation Window: {params['est_window']} days",
            f"Rebalance Every:   {params['rebal_freq']} days",
            f"Risk Aversion λ:   {params['risk_aversion']}",
            f"Stop-Loss:         {params['stop_loss_pct']}%",
            f"Cooldown:          {params['cooldown_days']} days",
        ]
        zf.writestr("parameters.txt", "\n".join(param_lines))

    zip_buffer.seek(0)

    any_selected = any(st.session_state[k] for k in
                       ["dl_metrics", "dl_cumret", "dl_weights", "dl_signals", "dl_zscores"])

    if any_selected:
        st.download_button(
            label="⬇ Download Selected Files as ZIP",
            data=zip_buffer,
            file_name="smart_mining_export.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.warning("Please select at least one file to download.")
