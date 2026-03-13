"""
Data Module — Smart Mining Project
Downloads and cleans ETF price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ASSETS = {
    "CPER": "Copper",
    "LIT":  "Lithium",
    "JJN":  "Nickel",
    "JJU":  "Aluminum",
    "SLV":  "Silver",
    "PPLT": "Platinum",
    "PALL": "Palladium",
    "GLD":  "Gold (Benchmark)",
}

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.
    Returns a DataFrame with dates as index and tickers as columns.
    Missing values are forward-filled then back-filled.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # Drop columns that are entirely NaN (ticker not found)
    prices = prices.dropna(axis=1, how="all")

    # Forward-fill gaps (weekends, holidays), then back-fill leading NaNs
    prices = prices.ffill().bfill()

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def get_data_summary(prices: pd.DataFrame) -> dict:
    """Return summary statistics for the UI."""
    returns = compute_returns(prices)
    summary = {}
    for col in prices.columns:
        summary[col] = {
            "start_price": round(float(prices[col].iloc[0]), 2),
            "end_price":   round(float(prices[col].iloc[-1]), 2),
            "total_return": round(float((prices[col].iloc[-1] / prices[col].iloc[0] - 1) * 100), 2),
            "ann_vol":     round(float(returns[col].std() * np.sqrt(252) * 100), 2),
            "n_obs":       int(prices[col].notna().sum()),
        }
    return summary
