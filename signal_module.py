"""
Signal Module — Smart Mining Project
Mean Reversion signal using z-score of price vs. rolling mean.

Logic:
  z_score = (price - rolling_mean) / rolling_std
  z > +threshold  → SELL  (price above mean → expect reversion downward)
  z < -threshold  → BUY   (price below mean → expect reversion upward)
  else            → HOLD
"""

import pandas as pd
import numpy as np


def compute_zscore(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute rolling z-score for each asset.
    lookback: number of trading days for rolling window (default 20 = ~1 month).
    """
    rolling_mean = prices.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std  = prices.rolling(window=lookback, min_periods=lookback).std()
    zscore = (prices - rolling_mean) / rolling_std
    return zscore


def generate_signals(prices: pd.DataFrame, lookback: int = 20, threshold: float = 1.0) -> pd.DataFrame:
    """
    Generate discrete signals: +1 (BUY), -1 (SELL), 0 (HOLD).
    
    Parameters
    ----------
    prices    : DataFrame of daily prices
    lookback  : rolling window length in trading days
    threshold : z-score threshold to trigger signal

    Returns
    -------
    signals   : DataFrame with same shape as prices, values in {-1, 0, 1}
    """
    zscore = compute_zscore(prices, lookback)

    signals = pd.DataFrame(0, index=zscore.index, columns=zscore.columns)
    signals[zscore < -threshold] = 1   # BUY signal
    signals[zscore >  threshold] = -1  # SELL signal

    return signals


def compute_signal_strength(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Return raw z-scores (continuous) for visualization in the UI."""
    return compute_zscore(prices, lookback)
