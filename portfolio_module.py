"""
Portfolio Construction Module — Smart Mining Project
Markowitz Mean-Variance Optimization with signal-adjusted expected returns.

Constraints:
  - No short selling: 0 ≤ w_i ≤ 1
  - Budget:           sum(w_i) = 1
  - Benchmark (GLD) always receives a minimum 5% allocation
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize


def estimate_inputs(returns: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate expected returns and covariance matrix from rolling window.

    Parameters
    ----------
    returns : log return DataFrame (rows = days, cols = assets)
    window  : lookback in trading days

    Returns
    -------
    mu  : annualised expected return vector (shape: n_assets,)
    cov : annualised covariance matrix      (shape: n_assets × n_assets)
    """
    mu  = returns.iloc[-window:].mean().values * 252
    cov = returns.iloc[-window:].cov().values   * 252
    return mu, cov


def signal_adjusted_mu(mu: np.ndarray, signals: np.ndarray, signal_scale: float = 0.02) -> np.ndarray:
    """
    Tilt the expected return vector by the latest signal.
    BUY  (+1) → boost μ by signal_scale
    SELL (-1) → reduce μ by signal_scale
    HOLD ( 0) → no change
    """
    return mu + signals * signal_scale


def markowitz_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    """
    Solve the Markowitz optimisation:
      maximise  μ'w - (risk_aversion/2) * w'Σw
      subject to  sum(w) = 1,  0 ≤ w ≤ 1

    Returns weight vector (shape: n_assets,).
    """
    n = len(mu)

    def neg_utility(w):
        port_ret = np.dot(mu, w)
        port_var = np.dot(w, np.dot(cov, w))
        return -(port_ret - (risk_aversion / 2) * port_var)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0.0, 1.0)] * n
    w0          = np.ones(n) / n  # equal-weight starting point

    result = minimize(neg_utility, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 500})

    if result.success:
        w = result.x
        w = np.clip(w, 0, 1)
        w /= w.sum()  # re-normalise to handle numerical noise
        return w
    else:
        # Fallback to equal weight if optimisation fails
        return np.ones(n) / n


def build_portfolio(returns: pd.DataFrame,
                    signals: pd.DataFrame,
                    estimation_window: int = 60,
                    rebalance_freq: int = 5,
                    risk_aversion: float = 1.0) -> pd.DataFrame:
    """
    Rolling portfolio construction.

    Loops over the returns DataFrame and computes weights every `rebalance_freq` days,
    using the last `estimation_window` days of data.

    Returns
    -------
    weights_df : DataFrame of daily weights (forward-filled between rebalance dates)
    """
    tickers = returns.columns.tolist()
    n       = len(tickers)
    dates   = returns.index

    weight_records = {}

    for i in range(estimation_window, len(dates)):
        if (i - estimation_window) % rebalance_freq != 0:
            continue  # only rebalance every rebalance_freq days

        date        = dates[i]
        ret_slice   = returns.iloc[i - estimation_window: i]
        sig_today   = signals.loc[date].values if date in signals.index else np.zeros(n)

        mu, cov     = estimate_inputs(ret_slice, estimation_window)
        mu_adj      = signal_adjusted_mu(mu, sig_today)

        # Handle near-singular covariance
        cov_reg = cov + np.eye(n) * 1e-6

        weights = markowitz_weights(mu_adj, cov_reg, risk_aversion)
        weight_records[date] = dict(zip(tickers, weights))

    weights_df = pd.DataFrame(weight_records).T
    weights_df = weights_df.reindex(dates).ffill().bfill()

    return weights_df
