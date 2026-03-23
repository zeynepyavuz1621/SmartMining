"""
Risk Control Module — Smart Mining Project
Stop-Loss rule: when an asset hits its drawdown limit, its weight
flows into CASH (not redistributed to other metals).

This means the portfolio can go partially or fully to cash during
stress periods.
"""

import pandas as pd
import numpy as np
from portfolio_module import CASH


def apply_stop_loss(prices: pd.DataFrame,
                    weights: pd.DataFrame,
                    stop_loss_pct: float = 0.10,
                    cooldown_days: int = 5) -> pd.DataFrame:
    """
    Apply a stop-loss rule to portfolio weights.

    When an asset's price drops more than `stop_loss_pct` from its
    20-day rolling peak, its weight is set to 0 and moved to CASH.
    The asset then enters a cooldown period.

    Parameters
    ----------
    prices        : daily price DataFrame (metal tickers only, no CASH)
    weights       : daily weight DataFrame including CASH column
    stop_loss_pct : max allowed drawdown from 20-day peak (e.g. 0.10 = 10%)
    cooldown_days : days the asset stays out after stop-loss triggers

    Returns
    -------
    adjusted weights DataFrame
    """
    metal_tickers = [c for c in weights.columns if c != CASH]
    adj_w         = weights.copy()
    cooldowns     = {t: 0 for t in metal_tickers}

    for i in range(1, len(weights)):
        date = weights.index[i]
        row  = adj_w.loc[date].copy()

        freed_weight = 0.0   # weight released by stop-losses this day

        for t in metal_tickers:
            # Count down cooldown
            if cooldowns[t] > 0:
                cooldowns[t] -= 1
                freed_weight += row[t]
                row[t] = 0.0
                continue

            if t not in prices.columns:
                continue

            # Rolling 20-day peak
            window  = prices[t].iloc[max(0, i - 20): i + 1]
            peak    = window.max()
            if date not in prices.index:
                continue
            current  = prices.loc[date, t]
            drawdown = (current - peak) / peak   # negative number

            if drawdown < -stop_loss_pct:
                freed_weight  += row[t]
                row[t]         = 0.0
                cooldowns[t]   = cooldown_days

        # freed weight → mevcut ağırlıklara orantılı dağıt
        remaining = row.drop(index=[t for t in metal_tickers if cooldowns[t] > 0])
        other_total = remaining.sum()
        if other_total > 1e-8:
            row[remaining.index] += freed_weight * (remaining / other_total)
        else:
            row[CASH] += freed_weight

        # Normalise to ensure sum = 1 (handles floating-point drift)
        total = row.sum()
        if total > 1e-8:
            row = row / total
        else:
            row[CASH] = 1.0

        adj_w.loc[date] = row

    return adj_w


def compute_portfolio_drawdown(cum_returns: pd.Series) -> pd.Series:
    """Drawdown = (current - rolling max) / rolling max"""
    rolling_max = cum_returns.cummax()
    return (cum_returns - rolling_max) / rolling_max
