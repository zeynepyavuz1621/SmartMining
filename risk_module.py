"""
Risk Control Module — Smart Mining Project
Implements Stop-Loss rule at the portfolio level.

Stop-Loss logic:
  - Track a "high-water mark" for each asset.
  - If an asset's price drops more than `stop_loss_pct` from its rolling peak,
    set its weight to 0 and redistribute to other assets pro-rata.
  - A "cooldown" period prevents re-entry for `cooldown_days` after a stop-loss trigger.
"""

import pandas as pd
import numpy as np


def apply_stop_loss(prices: pd.DataFrame,
                    weights: pd.DataFrame,
                    stop_loss_pct: float = 0.10,
                    cooldown_days: int = 5) -> pd.DataFrame:
    """
    Apply a stop-loss rule to portfolio weights.

    Parameters
    ----------
    prices        : daily price DataFrame
    weights       : daily weight DataFrame (output of portfolio_module)
    stop_loss_pct : maximum allowed drawdown from rolling peak before exit (e.g. 0.10 = 10%)
    cooldown_days : number of days an asset is excluded after stop-loss triggers

    Returns
    -------
    adjusted_weights : weight DataFrame with stop-loss applied
    """
    tickers   = weights.columns.tolist()
    adj_w     = weights.copy()
    cooldowns = {t: 0 for t in tickers}  # days remaining in cooldown

    for i in range(1, len(weights)):
        date       = weights.index[i]
        prev_date  = weights.index[i - 1]
        row        = adj_w.loc[date].copy()

        for t in tickers:
            # Decrement cooldown counter
            if cooldowns[t] > 0:
                cooldowns[t] -= 1
                row[t] = 0.0
                continue

            if t not in prices.columns:
                continue

            # Check drawdown from peak over past 20 days
            window  = prices[t].iloc[max(0, i - 20): i + 1]
            peak    = window.max()
            current = prices[t].loc[date] if date in prices.index else peak

            drawdown = (current - peak) / peak  # negative number

            if drawdown < -stop_loss_pct:
                row[t]         = 0.0
                cooldowns[t]   = cooldown_days

        # Re-normalise remaining weights
        total = row.sum()
        if total > 1e-8:
            row = row / total
        else:
            row = pd.Series(1.0 / len(tickers), index=tickers)

        adj_w.loc[date] = row

    return adj_w


def compute_portfolio_drawdown(cum_returns: pd.Series) -> pd.Series:
    """
    Compute rolling drawdown series from a cumulative return series.
    Drawdown = (current value - rolling max) / rolling max
    """
    rolling_max = cum_returns.cummax()
    drawdown    = (cum_returns - rolling_max) / rolling_max
    return drawdown
