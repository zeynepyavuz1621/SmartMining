"""
Backtest & Evaluation Module — Smart Mining Project

Implements:
  - Rolling out-of-sample backtest
  - Benchmark: equal-weight (1/N) portfolio
  - Metrics: cumulative return, Sharpe ratio, max drawdown, Calmar ratio
"""

import pandas as pd
import numpy as np
from risk_module import compute_portfolio_drawdown


def run_backtest(prices: pd.DataFrame,
                 weights: pd.DataFrame) -> pd.Series:
    """
    Compute daily portfolio returns from prices and weights.

    Both DataFrames must share the same index and columns.
    Weights are applied with a 1-day lag (no look-ahead bias).

    Returns daily portfolio return series.
    """
    daily_returns = prices.pct_change(fill_method=None).dropna()

    # Align
    common_dates   = daily_returns.index.intersection(weights.index)
    daily_returns  = daily_returns.loc[common_dates]
    weights_lagged = weights.shift(1).loc[common_dates].fillna(0)

    # Clip stale NaN rows in weights
    weights_lagged = weights_lagged.ffill().fillna(0)

    # Portfolio return = sum(w_i * r_i)
    port_returns = (weights_lagged * daily_returns).sum(axis=1)
    return port_returns


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """Return daily returns of a 1/N equal-weight portfolio."""
    daily_returns = prices.pct_change().dropna()
    return daily_returns.mean(axis=1)


def compute_metrics(returns: pd.Series, rf: float = 0.0) -> dict:
    """
    Compute performance metrics from a daily return series.

    Parameters
    ----------
    returns : daily return Series
    rf      : daily risk-free rate (default 0)

    Returns
    -------
    dict of metrics
    """
    cum_ret   = (1 + returns).cumprod()
    total_ret = float(cum_ret.iloc[-1] - 1) * 100

    ann_ret   = float((1 + returns.mean()) ** 252 - 1) * 100
    ann_vol   = float(returns.std() * np.sqrt(252)) * 100

    excess    = returns - rf
    sharpe    = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    drawdown  = compute_portfolio_drawdown(cum_ret)
    max_dd    = float(drawdown.min()) * 100

    calmar    = ann_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    n_trades  = int((returns != 0).sum())

    return {
        "Total Return (%)":     round(total_ret, 2),
        "Ann. Return (%)":      round(ann_ret,   2),
        "Ann. Volatility (%)":  round(ann_vol,   2),
        "Sharpe Ratio":         round(sharpe,    3),
        "Max Drawdown (%)":     round(max_dd,    2),
        "Calmar Ratio":         round(calmar,    3),
        "Trading Days":         n_trades,
    }


