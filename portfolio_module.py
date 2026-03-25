"""
Portfolio Construction Module — Smart Mining Project
Markowitz Mean-Variance Optimisation with Cash as a safe-haven asset.

CASH is treated as a zero-return, zero-variance asset.
When no metal looks attractive the optimiser can allocate entirely to CASH.

Constraints:
  - No short selling: 0 ≤ w_i ≤ 1
  - Budget:           sum(w_i) = 1  (includes CASH)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

CASH = "__CASH__"


def add_cash(returns: pd.DataFrame) -> pd.DataFrame:
    """Append a CASH column with zero return every day."""
    r = returns.copy()
    r[CASH] = 0.0
    return r


def estimate_inputs(returns: pd.DataFrame, window: int):
    slice_ = returns.iloc[-window:]
    
    # Son satırda NaN olan sütunları düşür — o gün aktif olmayan varlıklar
    last_row = slice_.iloc[-1]
    active_cols = last_row.dropna().index.tolist()
    slice_ = slice_[active_cols]
    
    # Kalan NaN'ları da temizle
    slice_ = slice_.dropna()
    
    mu  = slice_.mean().values * 252
    cov = slice_.cov().values  * 252
    return mu, cov, slice_.columns.tolist()


def signal_adjusted_mu(mu: np.ndarray, signals: np.ndarray,
                       tickers: list, scale: float = 0.02) -> np.ndarray:
    """
    Tilt μ by the signal for non-CASH assets.
    BUY  (+1) → +scale
    SELL (-1) → -scale
    CASH      → untouched (0)
    """
    mu_adj = mu.copy()
    for i, t in enumerate(tickers):
        if t != CASH:
            mu_adj[i] += signals[i] * scale
    return mu_adj


def markowitz_weights(mu: np.ndarray, cov: np.ndarray,
                      risk_aversion: float = 1.0,
                      active_tickers: list = None) -> np.ndarray:
    """
    Maximise:  μ'w - (λ/2) w'Σw
    Subject to: sum(w) = 1,  0 ≤ w ≤ 1
    """
    n = len(mu)

    def neg_utility(w):
        return -(np.dot(mu, w) - (risk_aversion / 2) * np.dot(w, np.dot(cov, w)))

    result = minimize(
        neg_utility,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if result.success:
        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        return w
    else:
        # Fallback: put everything in CASH if available, otherwise equal weight
        w = np.zeros(n)
        if active_tickers and CASH in active_tickers:
            w[active_tickers.index(CASH)] = 1.0
        else:
            w = np.ones(n) / n
        return w


def build_portfolio(returns: pd.DataFrame,
                    signals: pd.DataFrame,
                    estimation_window: int = 60,
                    rebalance_freq: int = 5,
                    risk_aversion: float = 1.0) -> pd.DataFrame:
    """
    Rolling portfolio construction with CASH as an explicit asset.
    Delisted assets (NaN columns) are automatically excluded from
    the optimisation on each rebalance date.

    Returns a weight DataFrame whose columns are the original tickers + CASH.
    """
    returns_with_cash = add_cash(returns)
    all_tickers       = returns_with_cash.columns.tolist()
    dates             = returns_with_cash.index

    weight_records = {}

    for i in range(estimation_window, len(dates)):
        if (i - estimation_window) % rebalance_freq != 0:
            continue

        date      = dates[i]
        ret_slice = returns_with_cash.iloc[i - estimation_window: i]

        # Estimate inputs — NaN columns dropped inside
        mu, cov, active_tickers = estimate_inputs(ret_slice, estimation_window)

        # Signal vector for active tickers only
        sig_today = np.zeros(len(active_tickers))
        if date in signals.index:
            for j, t in enumerate(active_tickers):
                if t in signals.columns:
                    sig_today[j] = signals.loc[date, t]

        mu_adj  = signal_adjusted_mu(mu, sig_today, active_tickers)
        cov_reg = cov + np.eye(len(active_tickers)) * 1e-6
        weights_active = markowitz_weights(mu_adj, cov_reg, risk_aversion, active_tickers)

        # Map active weights back to full ticker list — inactive assets get 0
        full_weights = {t: 0.0 for t in all_tickers}
        for t, w in zip(active_tickers, weights_active):
            full_weights[t] = w

        weight_records[date] = full_weights

    weights_df = pd.DataFrame(weight_records).T
    weights_df = weights_df.reindex(dates).ffill().bfill()

    return weights_df