import yfinance as yf
for ticker in ["JJN", "JJU"]:
    df = yf.download(ticker, start="2015-01-01", end="2025-12-31", auto_adjust=True, progress=False)
    if not df.empty:
        print(f"{ticker}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} gün)")
    else:
        print(f"{ticker}: veri yok")

from data_module import download_prices, compute_returns
from portfolio_module import add_cash, estimate_inputs

prices = download_prices(
    ["CPER","LIT","JJN","JJU","SLV","PPLT","PALL","GLD"],
    "2021-01-04", "2025-12-31"
)
returns = compute_returns(prices)
returns_with_cash = add_cash(returns)

# 2023-07-28 sonrasındaki ilk işlem günü
after_delist = returns_with_cash.loc["2023-07-28":].index[0]
idx = returns_with_cash.index.get_loc(after_delist)
ret_slice = returns_with_cash.iloc[idx-60:idx]

mu, cov, active_tickers = estimate_inputs(ret_slice, 60)
print("Active tickers:", active_tickers)
print("mu:", dict(zip(active_tickers, mu.round(4))))