import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL"]
data = yf.download(tickers + ["SPY"], period="3y", auto_adjust=True)["Close"]

returns = data.pct_change()

portfolio = returns[tickers].mean(axis=1)
spy = returns["SPY"]

equity = (1 + portfolio).cumprod()
spy_equity = (1 + spy).cumprod()

print("Final return strategy:", equity.iloc[-1] - 1)
print("Final return SPY:", spy_equity.iloc[-1] - 1)

equity.to_csv("equity.csv")
