#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

UNIVERSES = {
    "top": ["NVDA","MSFT","AAPL","AMZN","META","GOOGL","AVGO","TSLA","COST","NFLX","AMD","ADBE","PEP","CSCO","QCOM","TXN","INTU","AMAT","BKNG","ISRG"],
    "tech": ["NVDA","MSFT","AAPL","AVGO","AMD","ADBE","CSCO","QCOM","TXN","AMAT","MU","PANW","CRWD","INTC","MRVL","SNPS","CDNS","ORCL","IBM"],
    "growth": ["TSLA","NFLX","META","AMZN","SHOP","DDOG","SNOW","MDB","PLTR","NET","ROKU","COIN","HOOD","DASH","UBER"],
    "healthcare": ["ISRG","REGN","VRTX","AMGN","GILD","MRNA","BIIB","DXCM","IDXX","ILMN"],
    "financials": ["PYPL","COIN","HOOD","SOFI","TROW","NDAQ","CME","IBKR"],
    "speculative": ["SOUN","IONQ","BBAI","RGTI","ACHR","JOBY","OPEN","DNA"],
}


def clean_float(x):
    try:
        if x is None or pd.isna(x) or math.isinf(float(x)):
            return None
        return float(x)
    except Exception:
        return None


def clamp(x, lo, hi):
    if x is None or pd.isna(x):
        return lo
    return max(lo, min(hi, float(x)))


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def spy_return(period):
    try:
        df = yf.download("SPY", period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        c = df["Close"].dropna()
        return float(c.iloc[-1] / c.iloc[0] - 1)
    except Exception:
        return 0.0


def fetch(ticker, period, spy_ret):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 40:
            print(f"[WARN] Not enough data: {ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        close = df["Close"].dropna()
        vol = df["Volume"].fillna(0)
        price = float(close.iloc[-1])
        change = (price / float(close.iloc[-2]) - 1) * 100
        ret = float(close.iloc[-1] / close.iloc[0] - 1)

        rsi_series = rsi(close)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        _, _, mh = macd(close)
        avg_vol20 = float(vol.tail(20).mean()) if float(vol.tail(20).mean()) else 1
        volume_ratio = float(vol.iloc[-1] / avg_vol20)

        info = {}
        try:
            info = yf.Ticker(ticker).get_info() or {}
        except Exception:
            pass

        pe = clean_float(info.get("trailingPE"))
        beta = clean_float(info.get("beta"))
        target = clean_float(info.get("targetMeanPrice"))
        upside = None if not target else (target / price - 1) * 100
        name = info.get("shortName") or info.get("longName") or ticker

        rel_strength = ret - spy_ret
        trend = 0
        if price > float(ema20.iloc[-1]): trend += 8
        if float(ema20.iloc[-1]) > float(ema50.iloc[-1]): trend += 8
        if not pd.isna(ema200.iloc[-1]) and float(ema50.iloc[-1]) > float(ema200.iloc[-1]): trend += 6

        r = clean_float(rsi_series.iloc[-1]) or 50
        rsi_score = 10 if r < 30 else 13 if r < 45 else 16 if r < 67 else 8 if r < 75 else 2
        rel_score = clamp((rel_strength + .20)/.55*22, 0, 22)
        vol_score = clamp((volume_ratio - .7)/2.3*10, 0, 10)
        macd_score = 8 if float(mh.iloc[-1]) > 0 else 2
        upside_score = 8 if upside is None else clamp((upside + 5)/45*12, 0, 12)
        pe_score = 5 if pe is None else clamp((90 - pe)/80*10, 0, 10)
        beta_score = 5 if beta is None else clamp((2.7 - beta)/2.2*8, 0, 8)

        score = clamp(trend + rsi_score + rel_score + vol_score + macd_score + upside_score + pe_score + beta_score, 0, 100)
        notes = []
        if beta and beta > 2.2: notes.append("High beta")
        if r > 76: notes.append("RSI extreme")
        if rel_strength > .10: notes.append("Strong vs SPY")
        if price < float(ema50.iloc[-1]): notes.append("Below EMA50")
        signal = "BUY" if score >= 72 else "HOLD" if score >= 52 else "SELL"

        return {
            "ticker": ticker,
            "company": name,
            "price": price,
            "change": change,
            "rsi": r,
            "relative_strength": rel_strength,
            "target_upside": upside,
            "beta": beta,
            "pe": pe,
            "volume_ratio": volume_ratio,
            "macd_hist": float(mh.iloc[-1]),
            "score": score,
            "signal": signal,
            "notes": notes or ["No major alerts"],
            "history": [{"date": str(i.date()), "close": float(v)} for i, v in close.tail(70).items()]
        }
    except Exception as e:
        print(f"[WARN] {ticker}: {e}")
        return None


def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    return obj


def build_html(payload):
    data = json.dumps(safe_json(payload))
    generated_at = payload["generated_at"]
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nasdaq Scout PRO</title>
<style>
body{{margin:0;background:#07111f;color:#e8f1ff;font-family:Arial,sans-serif}}header{{padding:24px;background:#0d1b2f;border-bottom:1px solid #20334f}}main{{padding:20px;max-width:1400px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}}.card{{background:#0d1b2f;border:1px solid #20334f;border-radius:16px;padding:16px}}.span3{{grid-column:span 3}}.span7{{grid-column:span 7}}.span5{{grid-column:span 5}}.span12{{grid-column:span 12}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #20334f;text-align:left;white-space:nowrap}}th{{color:#92a4bd}}select,input,button{{background:#09182a;color:#e8f1ff;border:1px solid #20334f;border-radius:10px;padding:9px}}.metric{{font-size:28px;font-weight:bold}}.muted{{color:#92a4bd}}.buy{{color:#2ce69b}}.hold{{color:#ffd166}}.sell{{color:#ff5d73}}canvas{{width:100%;height:260px;background:#071426;border-radius:12px}}@media(max-width:900px){{.span3,.span5,.span7{{grid-column:span 12}}}}
</style></head><body>
<header><h1>Nasdaq Scout PRO</h1><div class="muted">Generated {generated_at} - yfinance/Yahoo Finance - prototype only</div></header>
<main><div class="grid">
<section class="card span12"><select id="universe" onchange="render()"></select> <input id="search" placeholder="Search" oninput="render()"> <button onclick="downloadCSV()">Export CSV</button></section>
<section class="card span3"><div class="muted">Best candidate</div><div class="metric" id="best">-</div></section>
<section class="card span3"><div class="muted">Average score</div><div class="metric" id="avg">-</div></section>
<section class="card span3"><div class="muted">Buy signals</div><div class="metric buy" id="buy">-</div></section>
<section class="card span3"><div class="muted">Risk alerts</div><div class="metric hold" id="risk">-</div></section>
<section class="card span7"><h2>Ranking</h2><div style="overflow:auto"><table><thead><tr><th>#</th><th>Ticker</th><th>Company</th><th>Score</th><th>Signal</th><th>Price</th><th>1D</th><th>RSI</th><th>RelStr</th><th>Upside</th><th>Beta</th><th>Notes</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="card span5"><h2>Stock cockpit</h2><div id="cockpit" class="muted">Select a row.</div><canvas id="chart" width="700" height="280"></canvas></section>
<section class="card span12 muted">Composite score: trend + RSI + relative strength vs SPY + volume + MACD + upside + valuation + beta. Screening heuristic, not financial advice.</section>
</div></main>
<script>
const payload={data};
let current=Object.keys(payload.rankings)[0];
function f(x,d=2){{return x===null||x===undefined||Number.isNaN(x)?"-":Number(x).toFixed(d)}}
function init(){{const s=document.getElementById("universe");Object.keys(payload.rankings).forEach(u=>s.innerHTML+=`<option value="${{u}}">${{u}}</option>`);render();}}
function render(){{current=document.getElementById("universe").value||current;let q=(document.getElementById("search").value||"").toLowerCase();let rows=(payload.rankings[current]||[]).filter(x=>(x.ticker+x.company).toLowerCase().includes(q));document.getElementById("rows").innerHTML=rows.map((s,i)=>`<tr onclick="selectStock('${{s.ticker}}')"><td>${{i+1}}</td><td><b>${{s.ticker}}</b></td><td>${{s.company}}</td><td><b>${{f(s.score,1)}}</b></td><td class="${{String(s.signal).toLowerCase()}}">${{s.signal}}</td><td>$${{f(s.price,2)}}</td><td class="${{s.change>=0?'buy':'sell'}}">${{f(s.change,2)}}%</td><td>${{f(s.rsi,1)}}</td><td>${{f(s.relative_strength*100,1)}}%</td><td>${{f(s.target_upside,1)}}%</td><td>${{f(s.beta,2)}}</td><td>${{(s.notes||[]).join(', ')}}</td></tr>`).join("");let best=rows[0];document.getElementById("best").textContent=best?best.ticker:"-";document.getElementById("avg").textContent=f(rows.reduce((a,b)=>a+(b.score||0),0)/(rows.length||1),1);document.getElementById("buy").textContent=rows.filter(s=>s.signal==="BUY").length;document.getElementById("risk").textContent=rows.filter(s=>(s.notes||[]).some(n=>/risk|beta|extreme|Below/i.test(n))).length;if(best)selectStock(best.ticker);}}
function find(t){{for(const r of Object.values(payload.rankings)){{let s=r.find(x=>x.ticker===t);if(s)return s;}}}}
function selectStock(t){{let s=find(t);document.getElementById("cockpit").innerHTML=`<h3>${{s.ticker}} - ${{s.company}}</h3><p>Signal: <b class="${{String(s.signal).toLowerCase()}}">${{s.signal}}</b> - Score: ${{f(s.score,1)}} - Price: $${{f(s.price,2)}}</p><p>RSI ${{f(s.rsi,1)}} - Relative strength ${{f(s.relative_strength*100,1)}}% - Volume ratio ${{f(s.volume_ratio,2)}}x - MACD hist ${{f(s.macd_hist,3)}}</p>`;draw((s.history||[]).map(x=>x.close));}}
function draw(vals){{let c=document.getElementById("chart"),ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);if(!vals.length)return;let min=Math.min(...vals),max=Math.max(...vals);ctx.strokeStyle="#46d9a1";ctx.lineWidth=3;ctx.beginPath();vals.forEach((v,i)=>{{let x=i/(vals.length-1)*w,y=h-((v-min)/(max-min||1))*h*.82-h*.08;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}});ctx.stroke();}}
function downloadCSV(){{let rows=payload.rankings[current]||[],head=["ticker","company","price","change","rsi","relative_strength","target_upside","beta","pe","score","signal","notes"];let csv=[head.join(","),...rows.map(s=>head.map(k=>k==="notes"?(s.notes||[]).join("|"):(s[k]??"")).join(","))].join("\\n");let a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));a.download=current+"_ranking.csv";a.click();}}
init();
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="top")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--period", default="6mo")
    args = ap.parse_args()

    selected = UNIVERSES if args.all else {args.universe: UNIVERSES.get(args.universe, UNIVERSES["top"])}
    if args.quick:
        selected = {k: v[:8] for k, v in selected.items()}

    sr = spy_return(args.period)
    all_tickers = sorted(set(t for group in selected.values() for t in group))
    rows = []
    for i, t in enumerate(all_tickers, 1):
        print(f"[{i}/{len(all_tickers)}] Fetching {t}")
        item = fetch(t, args.period, sr)
        if item:
            rows.append(item)

    by_ticker = {r["ticker"]: r for r in rows}
    rankings = {}
    for name, tickers in selected.items():
        rankings[name] = sorted([by_ticker[t] for t in tickers if t in by_ticker], key=lambda x: x["score"], reverse=True)

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "rankings": rankings}
    (OUT / "latest_data.json").write_text(json.dumps(safe_json(payload), indent=2), encoding="utf-8")
    (OUT / "index.html").write_text(build_html(payload), encoding="utf-8")

    hist = []
    for universe, items in rankings.items():
        if items:
            b = items[0]
            hist.append({"generated_at": payload["generated_at"], "universe": universe, "ticker": b["ticker"], "company": b["company"], "price": b["price"], "score": b["score"], "signal": b["signal"]})
    pd.DataFrame(hist).to_excel(OUT / "HistorialScores.xlsx", index=False)

    print("Done.")
    print("Created output/index.html, output/latest_data.json, output/HistorialScores.xlsx")


if __name__ == "__main__":
    main()
