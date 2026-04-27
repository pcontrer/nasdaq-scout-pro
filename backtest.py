import yfinance as yf
import pandas as pd
import numpy as np

UNIVERSES = {
    "top": ["NVDA","MSFT","AAPL","AMZN","META","GOOGL"],
    "tech": ["NVDA","MSFT","AAPL","AVGO","AMD","ADBE","CSCO","QCOM"],
    "growth": ["TSLA","NFLX","META","AMZN","SHOP","PLTR","COIN","UBER"]
}

TOP_N_LIST = [2, 3, 5]
YEARS_LIST = [3, 5, 7]


def calculate_rsi(px, period=14):
    delta = px.diff()
    gain = delta.clip(lower=0).ewm(span=period).mean()
    loss = -delta.clip(upper=0).ewm(span=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_backtest(tickers, years, top_n):
    data = yf.download(tickers + ["SPY"], period=f"{years}y", auto_adjust=True)["Close"]

    equity = 1
    spy_equity = 1

    rebalance_dates = data.resample("ME").last().index

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
                rsi = calculate_rsi(px).iloc[-1]
                momentum = px.iloc[-1] / px.iloc[-63] - 1

                if px.iloc[-1] > ema50 and rsi < 70:
                    selected.append((t, momentum))

            except Exception:
                continue

        selected = sorted(selected, key=lambda x: x[1], reverse=True)[:top_n]

        if not selected:
            continue

        period_returns = []

        for t, _ in selected:
            px = data[t].loc[date:next_date].dropna()

            if len(px) < 2:
                continue

            ret = px.iloc[-1] / px.iloc[0] - 1

            min_drawdown = (px / px.iloc[0] - 1).min()
            if min_drawdown < -0.10:
                ret = -0.10

            period_returns.append(ret)

        portfolio_return = np.mean(period_returns) if period_returns else 0

        spy_px = data["SPY"].loc[date:next_date].dropna()
        spy_ret = spy_px.iloc[-1] / spy_px.iloc[0] - 1

        equity *= (1 + portfolio_return)
        spy_equity *= (1 + spy_ret)

    return {
        "strategy_total_return": equity - 1,
        "spy_total_return": spy_equity - 1,
        "alpha_vs_spy": equity - spy_equity
    }


results = []

for universe_name, tickers in UNIVERSES.items():
    for years in YEARS_LIST:
        for top_n in TOP_N_LIST:
            print(f"Running: universe={universe_name}, years={years}, top_n={top_n}")
            result = run_backtest(tickers, years, top_n)

            results.append({
                "universe": universe_name,
                "years": years,
                "top_n": top_n,
                **result
            })

df = pd.DataFrame(results)

df.to_csv("sensitivity_results.csv", index=False)

summary = df.sort_values("alpha_vs_spy", ascending=False)
summary.to_csv("sensitivity_summary.csv", index=False)

print("\nSENSITIVITY RESULTS\n")
print(summary.to_string(index=False))
