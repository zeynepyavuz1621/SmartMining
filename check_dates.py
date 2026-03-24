import yfinance as yf
for ticker in ["JJN", "JJU"]:
    df = yf.download(ticker, start="2015-01-01", end="2025-12-31", auto_adjust=True, progress=False)
    if not df.empty:
        print(f"{ticker}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} gün)")
    else:
        print(f"{ticker}: veri yok")