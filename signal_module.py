"""
Signal Module — Smart Mining Project
Mean Reversion signal using z-score of price vs. rolling mean.

Threshold is fixed at 1.645 (90% confidence interval of normal distribution).
  z > +1.645  → SELL  (price statistically too high → expect downward reversion)
  z < -1.645  → BUY   (price statistically too low  → expect upward reversion)
  else        → HOLD
"""

import pandas as pd
import numpy as np

Z_THRESHOLD = 1.645   # 90% confidence interval — not exposed to user


def compute_zscore(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rolling z-score for each asset."""
    rolling_mean = prices.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std  = prices.rolling(window=lookback, min_periods=lookback).std()
    return (prices - rolling_mean) / rolling_std


def generate_signals(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Generate discrete signals: +1 (BUY), -1 (SELL), 0 (HOLD).
    Threshold is fixed at Z_THRESHOLD = 1.645 (90% CI).
    """
    zscore  = compute_zscore(prices, lookback)
    signals = pd.DataFrame(0, index=zscore.index, columns=zscore.columns)
    signals[zscore < -Z_THRESHOLD] =  1
    signals[zscore >  Z_THRESHOLD] = -1
    return signals


def compute_signal_strength(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Return raw z-scores for visualisation."""
    return compute_zscore(prices, lookback)
