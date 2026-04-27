import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL"]
data = yf.download(tickers + ["SPY"], period="5y", auto_adjust=True)["Close"]

returns = data.pct_change()

equity = 1
spy_equity = 1

portfolio_curve = []
spy_curve = []

rebalance_dates = data.resample("M").last().index

for i in range(3, len(rebalance_dates)-1):

    date = rebalance_dates[i]
    next_date = rebalance_dates[i+1]

    past_data = data.loc[:date]

    selected = []

    for t in tickers:
        try:
            px = past_data[t].dropna()

            if len(px) < 100:
                continue

            ema50 = px.ewm(span=50).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (px.diff().clip(lower=0).ewm(span=14).mean() /
                                     -px.diff().clip(upper=0).ewm(span=14).mean())))

            rsi_val = rsi.iloc[-1]

            momentum = px.iloc[-1] / px.iloc[-63] - 1

            if px.iloc[-1] > ema50 and rsi_val < 70:
                selected.append((t, momentum))

        except:
            continue

    selected = sorted(selected, key=lambda x: x[1], reverse=True)[:3]

    if not selected:
        continue

    period_returns = []

    for t, _ in selected:
        px = data[t].loc[date:next_date].dropna()

        if len(px) < 2:
            continue

        ret = px.iloc[-1] / px.iloc[0] - 1

        # STOP LOSS
        min_drawdown = (px / px.iloc[0] - 1).min()
        if min_drawdown < -0.10:
            ret = -0.10

        period_returns.append(ret)

    if period_returns:
        portfolio_return = np.mean(period_returns)
    else:
        portfolio_return = 0

    spy_px = data["SPY"].loc[date:next_date].dropna()
    spy_ret = spy_px.iloc[-1] / spy_px.iloc[0] - 1

    equity *= (1 + portfolio_return)
    spy_equity *= (1 + spy_ret)

    portfolio_curve.append(equity)
    spy_curve.append(spy_equity)

# RESULTADOS

print("\nRESULTADOS\n")

print("Strategy return:", round(equity - 1, 3))
print("SPY return:", round(spy_equity - 1, 3))

df = pd.DataFrame({
    "date": rebalance_dates[3:3+len(portfolio_curve)],
    "strategy": portfolio_curve,
    "spy": spy_curve,
    "strategy_return": np.array(portfolio_curve) - 1,
    "spy_return": np.array(spy_curve) - 1,
    "alpha_vs_spy": np.array(portfolio_curve) - np.array(spy_curve)
})

df.to_csv("equity.csv", index=False)

summary = pd.DataFrame([{
    "final_strategy_value": equity,
    "final_spy_value": spy_equity,
    "strategy_total_return": equity - 1,
    "spy_total_return": spy_equity - 1,
    "alpha_vs_spy": equity - spy_equity
}])

summary.to_csv("summary.csv", index=False)

print("\nSUMMARY\n")
print(summary.to_string(index=False))
