"""
experiments.py — Smart Mining Deney Koşturucu
YAP 471 · Team MetalMinds

Bölüm 1: Tek varlık analizleri   (8 varlık × 2 vade = 16 deney)
Bölüm 2: Portföy karşılaştırmaları (4 portföy)
Bölüm 3: Parametre duyarlılık testleri
Bölüm 4: Kriz dönemi testleri

Çıktı: experiments_output/ klasörüne CSV + Plotly HTML
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data_module      import download_prices, compute_returns, ASSETS
from signal_module    import generate_signals
from portfolio_module import build_portfolio, CASH
from risk_module      import apply_stop_loss, compute_portfolio_drawdown
from backtest_module  import run_backtest, equal_weight_benchmark, compute_metrics

# ─── Çıktı klasörü ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_output")
os.makedirs(OUT, exist_ok=True)

# ─── Renk paleti ────────────────────────────────────────────────────────────
PALETTE = ["#e6a817","#58a6ff","#3fb950","#f85149","#d2a8ff",
           "#ffa657","#79c0ff","#56d364","#ff7b72","#bc8cff"]
DARK_BG = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", family="IBM Plex Mono"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    xaxis=dict(gridcolor="#21262d"),
)

# ─── Varsayılan parametreler ─────────────────────────────────────────────────
DEFAULT = dict(lookback=20, est_window=60, rebal_freq=5,
               risk_aversion=1.0, stop_loss_pct=0.10, cooldown_days=5)

ALL_TICKERS  = list(ASSETS.keys())  # 8 varlık

# ─── Yardımcı: tam pipeline ──────────────────────────────────────────────────
def run_pipeline(tickers, start, end, **kwargs):
    """
    Veri indir → sinyal → portföy → stop-loss → backtest
    Returns: (port_returns, bmark_returns, port_metrics, bmark_metrics, weights)
    """
    p = dict(DEFAULT); p.update(kwargs)

    prices = download_prices(tickers, start, end)
    if prices.empty or len(prices) < p["est_window"] + 30:
        return None

    available = [t for t in tickers if t in prices.columns]
    prices    = prices[available]
    returns   = compute_returns(prices)

    signals      = generate_signals(prices, lookback=p["lookback"])
    weights_raw  = build_portfolio(returns, signals,
                                   estimation_window=p["est_window"],
                                   rebalance_freq=p["rebal_freq"],
                                   risk_aversion=p["risk_aversion"])
    weights      = apply_stop_loss(prices, weights_raw,
                                   stop_loss_pct=p["stop_loss_pct"],
                                   cooldown_days=p["cooldown_days"])

    prices_bt       = prices.copy(); prices_bt[CASH] = 1.0
    port_returns    = run_backtest(prices_bt, weights)
    bmark_returns   = equal_weight_benchmark(prices)

    common          = port_returns.index.intersection(bmark_returns.index)
    port_returns    = port_returns.loc[common]
    bmark_returns   = bmark_returns.loc[common]

    return dict(
        port_returns=port_returns,
        bmark_returns=bmark_returns,
        port_metrics=compute_metrics(port_returns),
        bmark_metrics=compute_metrics(bmark_returns),
        weights=weights,
        prices=prices,
    )

def save_fig(fig, filename):
    path = os.path.join(OUT, filename)
    fig.write_html(path)
    print(f"  → {filename}")

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — Tek Varlık Analizleri
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BÖLÜM 1 — Tek Varlık Analizleri (8 varlık × 2 vade)")
print("="*60)

DATE_RANGES = {
    "short_3y": ("2023-01-01", "2025-12-31"),
    "long_5y":  ("2021-01-04", "2025-12-31"),
}

sec1_rows = []

for period_name, (start, end) in DATE_RANGES.items():
    print(f"\n  Vade: {period_name} ({start} → {end})")
    figs_cum = []
    for ticker in ALL_TICKERS:
        print(f"    {ticker}...", end=" ", flush=True)
        res = run_pipeline([ticker], start, end)
        if res is None:
            print("VERİ YOK")
            continue

        pm = res["port_metrics"]
        bm = res["bmark_metrics"]

        # Cash yüzdesi
        cash_col = res["weights"][CASH] if CASH in res["weights"].columns else pd.Series(0.0, index=res["weights"].index)
        cash_pct = float(cash_col.mean() * 100)

        sec1_rows.append({
            "period": period_name, "ticker": ticker, "asset": ASSETS[ticker],
            "strategy_return": pm["Total Return (%)"],
            "benchmark_return": bm["Total Return (%)"],
            "excess_return": pm["Total Return (%)"] - bm["Total Return (%)"],
            "sharpe": pm["Sharpe Ratio"],
            "max_drawdown": pm["Max Drawdown (%)"],
            "ann_vol": pm["Ann. Volatility (%)"],
            "cash_pct": round(cash_pct, 1),
        })
        print(f"Ret={pm['Total Return (%)']:.1f}%  Sharpe={pm['Sharpe Ratio']:.3f}  DD={pm['Max Drawdown (%)']:.1f}%")

    # Tek vadedeki tüm varlıkları bir grafikte karşılaştır
    period_data = [r for r in sec1_rows if r["period"] == period_name]
    if period_data:
        df_p = pd.DataFrame(period_data).sort_values("excess_return", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Strategy", x=df_p["ticker"], y=df_p["strategy_return"],
                             marker_color=PALETTE[0], text=[f"{v:.1f}%" for v in df_p["strategy_return"]],
                             textposition="outside", textfont=dict(color="#c9d1d9")))
        fig.add_trace(go.Bar(name="Benchmark (Hold)", x=df_p["ticker"], y=df_p["benchmark_return"],
                             marker_color=PALETTE[1]))
        fig.update_layout(**DARK_BG, barmode="group", height=420,
                          title=f"Bölüm 1 — Toplam Getiri: {period_name}",
                          yaxis=dict(gridcolor="#21262d", title_text="Getiri (%)"))
        save_fig(fig, f"s1_{period_name}_returns.html")

        fig2 = make_subplots(rows=1, cols=3,
                             subplot_titles=["Sharpe Ratio","Max Drawdown (%)","Cash %"])
        fig2.add_trace(go.Bar(x=df_p["ticker"], y=df_p["sharpe"],
                              marker_color=PALETTE[2], showlegend=False), row=1, col=1)
        fig2.add_trace(go.Bar(x=df_p["ticker"], y=df_p["max_drawdown"].abs(),
                              marker_color=PALETTE[3], showlegend=False), row=1, col=2)
        fig2.add_trace(go.Bar(x=df_p["ticker"], y=df_p["cash_pct"],
                              marker_color=PALETTE[4], showlegend=False), row=1, col=3)
        fig2.update_layout(**DARK_BG, height=380,
                           title=f"Bölüm 1 — Risk Metrikleri: {period_name}")
        for col_i in [1,2,3]:
            fig2.update_xaxes(gridcolor="#21262d", row=1, col=col_i)
            fig2.update_yaxes(gridcolor="#21262d", row=1, col=col_i)
        save_fig(fig2, f"s1_{period_name}_risk.html")

sec1_df = pd.DataFrame(sec1_rows)
sec1_df.to_csv(os.path.join(OUT, "section1_single_asset.csv"), index=False)
print(f"\n  CSV kaydedildi: section1_single_asset.csv")
print(sec1_df.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — Portföy Karşılaştırmaları
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BÖLÜM 2 — Portföy Karşılaştırmaları (4 portföy)")
print("="*60)

PORTFOLIOS = {
    "Full (8 varlık)":    ["CPER","LIT","JJN","JJU","SLV","PPLT","PALL","GLD"],
    "Industrial Metals":  ["CPER","LIT","JJN","JJU"],
    "Precious Metals":    ["SLV","PPLT","PALL","GLD"],
    "AI Core":            ["CPER","LIT","SLV"],
}

START2, END2 = "2021-01-04", "2025-12-31"
sec2_rows = []
fig_cum = go.Figure()
fig_dd  = go.Figure()

for i, (pname, tickers) in enumerate(PORTFOLIOS.items()):
    print(f"\n  {pname} {tickers}...", end=" ", flush=True)
    res = run_pipeline(tickers, START2, END2)
    if res is None:
        print("VERİ YOK"); continue

    pm = res["port_metrics"]
    bm = res["bmark_metrics"]
    cash_pct = float(res["weights"][CASH].mean() * 100) if CASH in res["weights"].columns else 0.0

    sec2_rows.append({"portfolio": pname, "assets": ",".join(tickers), **pm,
                      "cash_pct": round(cash_pct, 1)})
    print(f"Ret={pm['Total Return (%)']:.1f}%  Sharpe={pm['Sharpe Ratio']:.3f}  DD={pm['Max Drawdown (%)']:.1f}%")

    cum = (1 + res["port_returns"]).cumprod()
    fig_cum.add_trace(go.Scatter(x=cum.index, y=cum.values, name=pname,
                                 line=dict(color=PALETTE[i], width=2)))
    dd = compute_portfolio_drawdown(cum) * 100
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=pname,
                                line=dict(color=PALETTE[i], width=1.8),
                                fill="tozeroy",
                                fillcolor=f"rgba({int(PALETTE[i][1:3],16)},{int(PALETTE[i][3:5],16)},{int(PALETTE[i][5:7],16)},0.1)"))

# Equal-weight benchmark (full 8 varlık)
prices_bm = download_prices(ALL_TICKERS, START2, END2)
bm_ret    = equal_weight_benchmark(prices_bm[[t for t in ALL_TICKERS if t in prices_bm.columns]])
cum_bm    = (1 + bm_ret).cumprod()
fig_cum.add_trace(go.Scatter(x=cum_bm.index, y=cum_bm.values, name="EW Benchmark",
                             line=dict(color="#444c56", width=1.5, dash="dot")))

fig_cum.update_layout(**DARK_BG, height=450,
                      title="Bölüm 2 — Kümülatif Getiri Karşılaştırması",
                      yaxis=dict(gridcolor="#21262d", title_text="Portföy Değeri (Normalised)"))
save_fig(fig_cum, "s2_portfolio_cumret.html")

fig_dd.update_layout(**DARK_BG, height=380,
                     title="Bölüm 2 — Drawdown Karşılaştırması",
                     yaxis=dict(gridcolor="#21262d", title_text="Drawdown (%)"))
save_fig(fig_dd, "s2_portfolio_drawdown.html")

sec2_df = pd.DataFrame(sec2_rows)
sec2_df.to_csv(os.path.join(OUT, "section2_portfolios.csv"), index=False)
print(f"\n  CSV kaydedildi: section2_portfolios.csv")
print(sec2_df[["portfolio","Total Return (%)","Sharpe Ratio","Max Drawdown (%)","cash_pct"]].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — Parametre Duyarlılık Testleri
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BÖLÜM 3 — Parametre Duyarlılık Testleri")
print("="*60)

START3, END3 = "2021-01-04", "2025-12-31"
TICKERS3 = ALL_TICKERS

def sensitivity_test(param_name, values, label, **fixed_kwargs):
    rows = []
    cum_fig = go.Figure()
    for i, val in enumerate(values):
        kwargs = dict(fixed_kwargs)
        kwargs[param_name] = val
        print(f"    {param_name}={val}...", end=" ", flush=True)
        res = run_pipeline(TICKERS3, START3, END3, **kwargs)
        if res is None:
            print("VERİ YOK"); continue
        pm = res["port_metrics"]
        rows.append({"param_value": val, **pm})
        print(f"Ret={pm['Total Return (%)']:.1f}%  Sharpe={pm['Sharpe Ratio']:.3f}  DD={pm['Max Drawdown (%)']:.1f}%")

        cum = (1 + res["port_returns"]).cumprod()
        cum_fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
                                     name=f"{label}={val}",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2)))

    cum_fig.update_layout(**DARK_BG, height=420,
                          title=f"Bölüm 3 — {param_name} Duyarlılığı: Kümülatif Getiri",
                          yaxis=dict(gridcolor="#21262d"))
    save_fig(cum_fig, f"s3_{param_name}_cumret.html")

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sharpe vs param scatter
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df["param_value"], y=df["Sharpe Ratio"],
                                   mode="lines+markers",
                                   line=dict(color=PALETTE[0], width=2),
                                   marker=dict(size=10, color=PALETTE[0])))
        bg_no_xaxis = {k: v for k, v in DARK_BG.items() if k != "xaxis"}
        fig_s.update_layout(**bg_no_xaxis, height=320,
                            title=f"Bölüm 3 — {param_name} vs Sharpe Ratio",
                            xaxis=dict(gridcolor="#21262d", title_text=label),
                            yaxis=dict(gridcolor="#21262d", title_text="Sharpe Ratio"))
        save_fig(fig_s, f"s3_{param_name}_sharpe.html")

        fig_dd2 = go.Figure()
        fig_dd2.add_trace(go.Scatter(x=df["param_value"], y=df["Max Drawdown (%)"].abs(),
                                     mode="lines+markers",
                                     line=dict(color=PALETTE[3], width=2),
                                     marker=dict(size=10, color=PALETTE[3])))
        fig_dd2.update_layout(**bg_no_xaxis, height=320,
                              title=f"Bölüm 3 — {param_name} vs Max Drawdown",
                              xaxis=dict(gridcolor="#21262d", title_text=label),
                              yaxis=dict(gridcolor="#21262d", title_text="|Max Drawdown| (%)"))
        save_fig(fig_dd2, f"s3_{param_name}_drawdown.html")

    return df

# 3a. Lookback etkisi
print("\n  3a. Lookback Penceresi (5, 10, 20, 40, 60 gün):")
df3a = sensitivity_test("lookback", [5, 10, 20, 40, 60], "Lookback")
df3a.to_csv(os.path.join(OUT, "section3a_lookback.csv"), index=False)
print(df3a[["param_value","Total Return (%)","Sharpe Ratio","Max Drawdown (%)"]].to_string(index=False))

# 3b. Stop-loss etkisi (None = stop-loss yok → çok büyük değer)
print("\n  3b. Stop-Loss (5%, 10%, 15%, 20%, yok):")
SL_VALS = [0.05, 0.10, 0.15, 0.20, 0.99]  # 0.99 ≈ stop-loss yok
SL_LABELS = ["5%", "10%", "15%", "20%", "Yok"]
rows3b = []
cum_fig3b = go.Figure()
for i, (sl, lbl) in enumerate(zip(SL_VALS, SL_LABELS)):
    print(f"    stop_loss={lbl}...", end=" ", flush=True)
    res = run_pipeline(TICKERS3, START3, END3, stop_loss_pct=sl)
    if res is None:
        print("VERİ YOK"); continue
    pm = res["port_metrics"]
    rows3b.append({"stop_loss_label": lbl, "stop_loss_pct": sl*100, **pm})
    print(f"Ret={pm['Total Return (%)']:.1f}%  Sharpe={pm['Sharpe Ratio']:.3f}  DD={pm['Max Drawdown (%)']:.1f}%")
    cum = (1 + res["port_returns"]).cumprod()
    cum_fig3b.add_trace(go.Scatter(x=cum.index, y=cum.values, name=f"SL={lbl}",
                                   line=dict(color=PALETTE[i], width=2)))

cum_fig3b.update_layout(**DARK_BG, height=420,
                        title="Bölüm 3 — Stop-Loss Duyarlılığı",
                        yaxis=dict(gridcolor="#21262d"))
save_fig(cum_fig3b, "s3_stoploss_cumret.html")
df3b = pd.DataFrame(rows3b)
df3b.to_csv(os.path.join(OUT, "section3b_stoploss.csv"), index=False)
print(df3b[["stop_loss_label","Total Return (%)","Sharpe Ratio","Max Drawdown (%)"]].to_string(index=False))

# 3c. Risk Aversion etkisi
print("\n  3c. Risk Aversion (λ = 0.5, 1.0, 2.0, 3.5):")
df3c = sensitivity_test("risk_aversion", [0.5, 1.0, 2.0, 3.5], "λ")
df3c.to_csv(os.path.join(OUT, "section3c_risk_aversion.csv"), index=False)
print(df3c[["param_value","Total Return (%)","Sharpe Ratio","Max Drawdown (%)","Ann. Volatility (%)"]].to_string(index=False))

# 3d. Rebalance frekansı
print("\n  3d. Rebalance Frekansı (1, 5, 10, 21 gün):")
df3d = sensitivity_test("rebal_freq", [1, 5, 10, 21], "Rebalance Freq")
df3d.to_csv(os.path.join(OUT, "section3d_rebal_freq.csv"), index=False)
print(df3d[["param_value","Total Return (%)","Sharpe Ratio","Max Drawdown (%)"]].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — Kriz Dönemi Testleri
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BÖLÜM 4 — Kriz Dönemi Testleri")
print("="*60)

CRISIS_PERIODS = {
    "Commodity Crash (2022)":    ("2022-03-01", "2022-09-30"),
    "AI Hype Rally (2023-H1)":   ("2023-01-01", "2023-06-30"),
    "Volatilite Dönemi (2024-H2)":("2024-06-01", "2024-12-31"),
}

sec4_rows = []
fig_crisis_cum = make_subplots(
    rows=1, cols=3,
    subplot_titles=list(CRISIS_PERIODS.keys()),
)

for col_idx, (crisis_name, (cs, ce)) in enumerate(CRISIS_PERIODS.items(), start=1):
    print(f"\n  {crisis_name} ({cs} → {ce})")
    res = run_pipeline(ALL_TICKERS, cs, ce)
    if res is None:
        print("  VERİ YOK"); continue

    pm = res["port_metrics"]
    bm = res["bmark_metrics"]
    cash_pct = float(res["weights"][CASH].mean() * 100) if CASH in res["weights"].columns else 0.0

    print(f"    Strategy: Ret={pm['Total Return (%)']:.1f}%  Sharpe={pm['Sharpe Ratio']:.3f}  DD={pm['Max Drawdown (%)']:.1f}%")
    print(f"    Benchmark: Ret={bm['Total Return (%)']:.1f}%  Sharpe={bm['Sharpe Ratio']:.3f}")

    sec4_rows.append({
        "crisis": crisis_name, "start": cs, "end": ce,
        "strat_return": pm["Total Return (%)"],
        "bench_return": bm["Total Return (%)"],
        "excess_return": pm["Total Return (%)"] - bm["Total Return (%)"],
        "sharpe": pm["Sharpe Ratio"],
        "max_drawdown": pm["Max Drawdown (%)"],
        "ann_vol": pm["Ann. Volatility (%)"],
        "cash_pct": round(cash_pct, 1),
    })

    cum  = (1 + res["port_returns"]).cumprod()
    cum_b = (1 + res["bmark_returns"]).cumprod()

    fig_crisis_cum.add_trace(
        go.Scatter(x=cum.index, y=cum.values, name="Strategy",
                   line=dict(color=PALETTE[0], width=2),
                   showlegend=(col_idx == 1)),
        row=1, col=col_idx
    )
    fig_crisis_cum.add_trace(
        go.Scatter(x=cum_b.index, y=cum_b.values, name="Benchmark",
                   line=dict(color=PALETTE[1], width=1.5, dash="dot"),
                   showlegend=(col_idx == 1)),
        row=1, col=col_idx
    )

fig_crisis_cum.update_layout(**DARK_BG, height=420,
                             title="Bölüm 4 — Kriz Dönemlerinde Kümülatif Getiri")
for c in [1, 2, 3]:
    fig_crisis_cum.update_xaxes(gridcolor="#21262d", row=1, col=c)
    fig_crisis_cum.update_yaxes(gridcolor="#21262d", row=1, col=c)
save_fig(fig_crisis_cum, "s4_crisis_cumret.html")

# Bar chart: strateji vs benchmark excess return per crisis
sec4_df = pd.DataFrame(sec4_rows)
if not sec4_df.empty:
    fig4b = go.Figure()
    fig4b.add_trace(go.Bar(name="Strategy",  x=sec4_df["crisis"], y=sec4_df["strat_return"],
                           marker_color=PALETTE[0], text=[f"{v:.1f}%" for v in sec4_df["strat_return"]],
                           textposition="outside", textfont=dict(color="#c9d1d9")))
    fig4b.add_trace(go.Bar(name="Benchmark", x=sec4_df["crisis"], y=sec4_df["bench_return"],
                           marker_color=PALETTE[1]))
    fig4b.update_layout(**DARK_BG, barmode="group", height=420,
                        title="Bölüm 4 — Kriz Dönemleri: Getiri Karşılaştırması",
                        yaxis=dict(gridcolor="#21262d", title_text="Toplam Getiri (%)"))
    save_fig(fig4b, "s4_crisis_bar.html")

    fig4c = go.Figure()
    fig4c.add_trace(go.Bar(x=sec4_df["crisis"], y=sec4_df["excess_return"],
                           marker_color=[PALETTE[2] if v >= 0 else PALETTE[3] for v in sec4_df["excess_return"]],
                           text=[f"{v:+.1f}%" for v in sec4_df["excess_return"]],
                           textposition="outside", textfont=dict(color="#c9d1d9")))
    fig4c.add_hline(y=0, line_color="#444c56")
    fig4c.update_layout(**DARK_BG, height=360,
                        title="Bölüm 4 — Excess Return (Strateji − Benchmark)",
                        yaxis=dict(gridcolor="#21262d", title_text="Excess Return (%)"))
    save_fig(fig4c, "s4_crisis_excess.html")

sec4_df.to_csv(os.path.join(OUT, "section4_crisis.csv"), index=False)
print(f"\n  CSV kaydedildi: section4_crisis.csv")
print(sec4_df.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# ÖZET RAPOR
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÖZET — Tüm deney sonuçları:")
print("="*60)

print("\n[Bölüm 1] En iyi Sharpe (5 yıl):")
if not sec1_df.empty:
    b1 = sec1_df[sec1_df["period"]=="long_5y"].nlargest(3, "sharpe")[["ticker","sharpe","strategy_return","max_drawdown","cash_pct"]]
    print(b1.to_string(index=False))

print("\n[Bölüm 2] Portföy sıralaması (Sharpe):")
if not sec2_df.empty:
    print(sec2_df[["portfolio","Sharpe Ratio","Total Return (%)","Max Drawdown (%)"]].sort_values("Sharpe Ratio",ascending=False).to_string(index=False))

print("\n[Bölüm 3a] Optimal lookback (Sharpe):")
if not df3a.empty:
    print(df3a.nlargest(1,"Sharpe Ratio")[["param_value","Sharpe Ratio","Total Return (%)"]].to_string(index=False))

print("\n[Bölüm 3b] Optimal stop-loss (Sharpe):")
if not df3b.empty:
    print(df3b.nlargest(1,"Sharpe Ratio")[["stop_loss_label","Sharpe Ratio","Total Return (%)","Max Drawdown (%)"]].to_string(index=False))

print("\n[Bölüm 3c] Risk aversion vs volatilite:")
if not df3c.empty:
    print(df3c[["param_value","Ann. Volatility (%)","Sharpe Ratio"]].to_string(index=False))

print("\n[Bölüm 3d] Optimal rebalance frekansı:")
if not df3d.empty:
    print(df3d.nlargest(1,"Sharpe Ratio")[["param_value","Sharpe Ratio","Total Return (%)"]].to_string(index=False))

print("\n[Bölüm 4] Kriz dönemleri excess return:")
if not sec4_df.empty:
    print(sec4_df[["crisis","excess_return","sharpe","cash_pct"]].to_string(index=False))

print(f"\n✓ Tüm çıktılar: {OUT}")
print("  HTML grafikler tarayıcıda açılabilir.")
