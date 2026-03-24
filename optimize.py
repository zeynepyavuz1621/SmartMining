"""
optimize.py — Smart Mining Grid Search
YAP 471 Computational Finance Term Project
Team MetalMinds

Kullanim:
    python optimize.py                          # tum varliklar, 2021-2025
    python optimize.py --tickers CPER SLV       # belirli varliklar
    python optimize.py --start 2022-01-01 --end 2022-12-31 --label kriz_2022
"""

import sys, os, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from data_module      import download_prices, compute_returns, ASSETS
from signal_module    import generate_signals, Z_THRESHOLD
from portfolio_module import build_portfolio, CASH
from risk_module      import apply_stop_loss
from backtest_module  import run_backtest, equal_weight_benchmark, compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Parametre araliklari
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "lookback":       [10, 20, 30],        # 5 çok gürültülü, 40 çok uzun
    "est_window":     [30, 60],            # 20 kovaryans için az, 120 çok uzun
    "rebal_freq":     [1, 5, 21],          # günlük, haftalık, aylık
    "risk_aversion":  [0.5, 1.0, 3.5],    # agresif, nötr, muhafazakar
    "stop_loss_pct":  [0.05, 0.10, 0.20], # sıkı, standart, gevşek
    "cooldown_days":  [1, 5, 10],         # kısa, orta, uzun
}

METRIC_COLS = [
    "Sharpe Ratio", "Total Return (%)", "Ann. Return (%)",
    "Ann. Volatility (%)", "Max Drawdown (%)", "Calmar Ratio", "Avg Cash (%)"
]

PARAM_COLS = list(PARAM_GRID.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Tek kombinasyon calistir
# ─────────────────────────────────────────────────────────────────────────────
def run_combination(prices, returns, params):
    try:
        signals     = generate_signals(prices, lookback=params["lookback"])
        weights_raw = build_portfolio(returns, signals,
                                      estimation_window=params["est_window"],
                                      rebalance_freq=params["rebal_freq"],
                                      risk_aversion=params["risk_aversion"])
        weights     = apply_stop_loss(prices, weights_raw,
                                      stop_loss_pct=params["stop_loss_pct"],
                                      cooldown_days=params["cooldown_days"])
        prices_bt       = prices.copy()
        prices_bt[CASH] = 1.0
        port_returns    = run_backtest(prices_bt, weights)
        metrics         = compute_metrics(port_returns)
        cash_col        = weights[CASH] if CASH in weights.columns else pd.Series(0.0, index=weights.index)
        metrics["Avg Cash (%)"] = round(float(cash_col.mean() * 100), 2)
        return {**params, **metrics}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=list(ASSETS.keys()))
    parser.add_argument("--start",   default="2021-01-04")
    parser.add_argument("--end",     default="2025-12-31")
    parser.add_argument("--label",   default="")
    args = parser.parse_args()

    label   = args.label or f"{'_'.join(args.tickers[:3])}_{args.start[:4]}_{args.end[:4]}"
    tickers = args.tickers

    print(f"\n{'='*60}")
    print(f"Smart Mining Grid Search")
    print(f"Varliklar : {', '.join(tickers)}")
    print(f"Donem     : {args.start} - {args.end}")
    print(f"{'='*60}\n")

    # Veri indir
    print("Veri indiriliyor...")
    prices    = download_prices(tickers, args.start, args.end)
    available = [t for t in tickers if t in prices.columns]
    prices    = prices[available]
    returns   = compute_returns(prices)

    if len(prices) < 90:
        print("Yeterli veri yok. Tarih araligini genisletin.")
        sys.exit(1)

    print(f"{len(available)} varlik, {len(prices)} gun yuklendi.\n")

    # Kombinasyonlari olustur
    keys   = PARAM_COLS
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f"{total} kombinasyon taranıyor...\n")

    start_time = datetime.now()
    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        if params["est_window"] <= params["lookback"]:
            continue
        if len(prices) < params["est_window"] + 30:
            continue

        result = run_combination(prices, returns, params)
        if result:
            results.append(result)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            elapsed = datetime.now() - start_time
            rate = (i + 1) / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            mins, secs = divmod(int(remaining), 60)
            print(f"\r  [{bar}] {pct:.0f}%  ({i+1}/{total})  ~{mins}dk {secs:02d}sn kaldi", end="", flush=True)

    print(f"\n\n{len(results)} gecerli kombinasyon degerlendirildi.\n")

    if not results:
        print("Hicbir kombinasyon calismadi.")
        sys.exit(1)

    df = pd.DataFrame(results)

    # Her metrik icin top 3
    rankings = {
        "Best Sharpe Ratio":    (df.sort_values("Sharpe Ratio",       ascending=False).head(3), "yuksek iyi"),
        "Best Total Return":    (df.sort_values("Total Return (%)",    ascending=False).head(3), "yuksek iyi"),
        "Min Max Drawdown":     (df.sort_values("Max Drawdown (%)",    ascending=True ).head(3), "dusuk iyi"),
        "Best Calmar Ratio":    (df.sort_values("Calmar Ratio",        ascending=False).head(3), "yuksek iyi"),
        "Min Volatility":       (df.sort_values("Ann. Volatility (%)", ascending=True ).head(3), "dusuk iyi"),
    }

    # Terminale Sharpe top3 yazdir
    top3 = rankings["Best Sharpe Ratio"][0]
    print("=" * 60)
    print("EN IYI 3 (Sharpe'a gore)")
    print("=" * 60)
    for i, (_, row) in enumerate(top3.iterrows()):
        print(f"\n  {i+1}.")
        for p in PARAM_COLS:
            val = row[p]
            if p == "stop_loss_pct":
                print(f"    {p:20s}: {val*100:.0f}%")
            else:
                print(f"    {p:20s}: {val}")
        for m in METRIC_COLS:
            print(f"    {m:25s}: {row[m]:.3f}")

    # Benchmark
    bmark_returns = equal_weight_benchmark(prices)
    bmark_metrics = compute_metrics(bmark_returns)
    print(f"\nBenchmark (1/N equal-weight):")
    for m in METRIC_COLS[:-1]:
        print(f"   {m:25s}: {bmark_metrics.get(m, 0):.3f}")

    # ── CSV kaydet — tek dosya, birden fazla tablo ────────────────────────────
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = f"results/{label}_top3_{timestamp}.csv"

    all_cols = ["Rank", "Criteria (sort)"] + PARAM_COLS + METRIC_COLS

    with open(out_path, "w", encoding="utf-8") as f:
        # Bilgi blogu
        f.write(f"Smart Mining Grid Search - {label}\n")
        f.write(f"Donem,{args.start},{args.end}\n")
        f.write(f"Varliklar,{', '.join(available)}\n")
        f.write(f"Toplam kombinasyon,{len(results)}\n")
        f.write(f"Z-Score threshold (sabit),{Z_THRESHOLD}\n")
        f.write("\n")

        # Benchmark blogu
        f.write("BENCHMARK (1/N Equal-Weight)\n")
        f.write(",".join(METRIC_COLS[:-1]) + "\n")
        bmark_vals = [str(round(bmark_metrics.get(m, 0), 3)) for m in METRIC_COLS[:-1]]
        f.write(",".join(bmark_vals) + "\n")
        f.write("\n")

        # Her ranking icin tablo
        for criteria, (top_df, note) in rankings.items():
            f.write(f"{criteria} ({note})\n")
            f.write(",".join(all_cols) + "\n")

            for i, (_, row) in enumerate(top_df.iterrows()):
                vals = [str(i + 1), criteria]
                for p in PARAM_COLS:
                    val = row[p]
                    vals.append(f"{val*100:.0f}%" if p == "stop_loss_pct" else str(val))
                for m in METRIC_COLS:
                    vals.append(str(round(row[m], 3)))
                f.write(",".join(vals) + "\n")

            f.write("\n")

    print(f"\nKaydedildi: {out_path}\n")


if __name__ == "__main__":
    main()