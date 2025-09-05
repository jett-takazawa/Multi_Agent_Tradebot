#!/usr/bin/env python3
import os, csv, json, io, time, datetime as dt, requests, pandas as pd
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
import requests
import pyarrow as pa, pyarrow.parquet as pq
from typing import List
from Data_Collect import save_signal
from openai import OpenAI

client = OpenAI(
    api_key= "GROK_API_KEY"
)



from testing_candles import fetch_recent_cand

# ── static config ──────────────────────────────────────────
RESOLUTION_MIN   = 15         # 0=daily, 5=5-min, etc.
BARS_PER_PROMPT  = 280

grok_key = ("xai-GROK_API_KEY")
if not grok_key:
    raise RuntimeError("GROK_API_KEY is not set or is empty")

# point at xAI's endpoint instead of openai.com
grok_client = OpenAI(
    api_key=grok_key,
    base_url="https://api.x.ai/v1",  # your xAI region
    timeout=4500,                     # ← raise from 600 s to 3600 s
    max_retries=2                     # optional: retry transient errors
)
CSV_FILE         = "tradelogtosheets.csv"
AV_KEY           = "av key"
KEY_FILE         = "Credentials.json"
SHEET_ID         = "sheet key"
DATA_DIR_CANDLES  = "data/candles" 
DATA_DICT = {}
FIELDNAMES = [
    "timestamp","symbol","trend","pattern",
    "entry","stop","exit","Odds_Score",
    "strength","time","freshness","trend_alignment", "News_Volatility"
]

SYSTEM_PROMPT = """
You are an advanced Supply-and-Demand Pattern Detector for equities, precisely identifying trade zones using strict breakout and odds enhancer criteria. Do your best to find a zone.

Allowed Output:
- JSON object with exactly the following keys:
  {
    "symbol": str,
    "trend": "uptrend" | "downtrend" | "sideways",
    "pattern": "DBR" | "RBR" | "RBD" | "DBD",
    "entry": float,          # Proximal line (entry price)
    "stop": float,           # Distal line (stop-loss price)
    "exit": float,           # Proximal line of next opposing zone (target)
    "Odds_Score": float,     # Score between 0 (low) and 1 (high)
    "odds_enhancers": {
      "strength": "Strong" | "Good" | "Poor",
      "time": "Strong" | "Good" | "Poor",
      "freshness": "Fresh" | "Poor" | "Violated",
      "trend_alignment": "With" | "Sideways" | "Counter"
    }
  }

Pattern Criteria:
- Demand Zones (valid only in an uptrend or sideways market):
  - Drop-Base-Rally (DBR)
  - Rally-Base-Rally (RBR)
- Supply Zones (valid only in a downtrend or sideways market):
  - Rally-Base-Drop (RBD)
  - Drop-Base-Drop (DBD)

Zone Identification:
- Begin analyzing from the most recent candle backward, focusing initially on identifying a significant breakout movement. This breakout candle typically defines or confirms your base pattern and the potential supply/demand zone. Such breakout movements usually occur within the first 20 candles from the current candle backward.
- Proximal Line (Entry Price): Highest candle body within the identified base.
- Distal Line (Stop-Loss Price):
  - DBR: Lowest wick among leg-in, base, and leg-out candles.
  - RBR: Lowest wick within the base.
  - RBD: Highest wick among leg-in, base, and leg-out candles.
  - DBD: Highest wick within the base and leg-out candle.
- **Always prioritize the most recently formed valid zones based on breakout identification criteria.**

Odds Enhancers represent the probability of this found zone being a successful trade. Always calculate them:

1. Strength (Zone Departure):
   - Strong: Price movement ≥ 2× zone height AND breaks previous structure (+2).
   - Good: Price movement ≥ 2× zone height OR breaks previous structure (+1).
   - Poor: Neither condition met (0).

2. Time at Base:
   - Strong: 1–3 candles (+1).
   - Good: 4–6 candles (+0.5).
   - Poor: >6 candles (0).

3. Freshness:
   - Fresh: Proximal line untouched (+2).
   - Poor: Price touched zone ≤50% penetration (+1).
   - Violated: Price exceeded distal line (0) — *then find a new zone*.

4. Trend Alignment:
   - With Trend: zone aligns with primary trend direction (+2).
   - Sideways: neutral market (+1).
   - Counter: opposite to trend (0).

5. Risk-Reward Ratio (must exceed 3:1):
   - Long (Demand): (Exit – Entry) / (Entry – Stop) > 3 → +1; >6 → +2.
   - Short (Supply): (Entry – Exit) / (Stop – Entry) > 3 → +1; >6 → +2.

Calculate Odds_Score as (sum of points) ÷ 9.

Exit Price:
- Always set at the proximal line of the next opposing zone.
- For Demand trades, exit must be above entry.
- For Supply trades, exit must be below entry.

Never explain yourself. Only respond with the JSON trade object above.

""".strip()

# ── helper funcs ───────────────────────────────────────────

def fetch_recent_ohlcv(symbol,
                       resolution_min,
                       n_bars) -> pd.DataFrame:
    """
    Pull `n_bars` most-recent intraday candles, save them as a Parquet
    file, and return them to the caller.
    """
    if resolution_min:
        interval = f"{resolution_min}min"
        url =(
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}"
            f"&interval={interval}"
            f"&outputsize=full"
            f"&datatype=csv"
            f"&apikey={AV_KEY}")
    else:
        url = (
    "https://www.alphavantage.co/query?"
    f"function=TIME_SERIES_DAILY_ADJUSTED"   # or TIME_SERIES_DAILY
    f"&symbol={symbol}"
    f"&outputsize=full"                   # 100 most-recent days; use full for full history
    f"&datatype=csv"
    f"&apikey={AV_KEY}"
        )

    csv_text = requests.get(url, timeout=15).text
    if "Thank you for using" in csv_text or csv_text.startswith("{"):
        raise RuntimeError("Alpha Vantage throttle/error:\n"+csv_text[:120])
    df = (pd.read_csv(io.StringIO(csv_text))
            .iloc[::-1]                 # oldest→newest
            .tail(n_bars)               # newest n_bars
            .rename(columns={
                "timestamp":"t","open":"o","high":"h",
                "low":"l","close":"c","volume":"v"})
            .reset_index(drop=True))
    df["t"] = pd.to_datetime(df["t"]).astype(int)//10**9
    # ---------- write 1 file per fetch ----------

    os.makedirs(DATA_DIR_CANDLES, exist_ok=True)
    today = dt.date.today().isoformat()
    path   = f"{DATA_DIR_CANDLES}/{symbol}_{interval}_{today}.parquet"

    df.to_parquet(path, index=False, compression="zstd")   # no append needed"""
    return df


def row_to_candle(r):
    ts = dt.datetime.utcfromtimestamp(r.t).isoformat()
    return f"{ts} O:{r.o:.2f} H:{r.h:.2f} L:{r.l:.2f} C:{r.c:.2f}"

def build_user_msg(df, symbol):
    label = f"{RESOLUTION_MIN}-min" if RESOLUTION_MIN else "daily"
    body  = "\n".join(row_to_candle(r) for _, r in df.iterrows())
    return f"Symbol {symbol}. Last {len(df)} candles ({label}):\n{body}\nClassify trend and respond per rules."

def flatten_trade(tr):
    enh = tr.pop("odds_enhancers", {}) or {}
    global DICTupdate
    DICTupdate = { 
        "timestamp":        tr.get("timestamp",""),
        "symbol":           tr.get("symbol",""),
        "decision":         "TRADE",
        "reason":           "null",
        "trend":            tr.get("trend",""),
        "pattern":          tr.get("pattern",""),
        "entry":            tr.get("entry",""),
        "stop":             tr.get("stop",""),
        "exit":             tr.get("exit",""),
        "Odds_Score":       tr.get("Odds_Score",""),
        "strength":         enh.get("strength",""),
        "time":             enh.get("time",""),
        "freshness":        enh.get("freshness",""),
        "trend_alignment":  enh.get("trend_alignment","")}
    global DATA_DICT
    DATA_DICT.update(DICTupdate)

    return {
        "timestamp":        tr.get("timestamp",""),
        "symbol":           tr.get("symbol",""),
        "trend":            tr.get("trend",""),
        "pattern":          tr.get("pattern",""),
        "entry":            tr.get("entry",""),
        "stop":             tr.get("stop",""),
        "exit":             tr.get("exit",""),
        "Odds_Score":       tr.get("Odds_Score",""),
        "strength":         enh.get("strength",""),
        "time":             enh.get("time",""),
        "freshness":        enh.get("freshness",""),
        "trend_alignment":  enh.get("trend_alignment",""),
        
    }
# ── the single callable function ───────────────────────────


def run_bot_new(symbol: str, vol_score):
    """One complete evaluation cycle for a single symbol."""
    # initialize row
    row = {c: None for c in FIELDNAMES}
    row.update({
        "symbol": symbol,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "News_Volatility": vol_score
    })

    # 1. Fetch candles
    try:
        candles = fetch_recent_ohlcv(symbol, RESOLUTION_MIN, BARS_PER_PROMPT)
    except Exception as err:
        print("⚠️  Candle fetch failed:", err)
        row.update({"decision": "NO_TRADE", "reason": "NO_DATA"})
        save_signal(row)
        return

    # 2. Query Grok
    try:
        resp = grok_client.chat.completions.create(
            model="grok-4-0709",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_msg(candles, symbol)},
            ],
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as err:
        print("⚠️  Grok call failed:", err)
        row.update({"decision": "NO_TRADE", "reason": "LLM_ERROR"})
        save_signal(row)
        return

    # 3. Interpret reply
    if reply.upper() == "NO_TRADE":
        row.update({"decision": "NO_TRADE", "reason": "NO_ZONE"})
    else:
        try:
            trade = json.loads(reply)
            enh   = trade.get("odds_enhancers", {}) or {}
            row.update({
                "decision":        "TRADE",
                "reason":          None,
                "trend":           trade.get("trend"),
                "pattern":         trade.get("pattern"),
                "entry":           trade.get("entry"),
                "stop":            trade.get("stop"),
                "exit":            trade.get("exit"),
                "Odds_Score":      trade.get("Odds_Score"),
                "strength":        enh.get("strength"),
                "time":            enh.get("time"),
                "freshness":       enh.get("freshness"),
                "trend_alignment": enh.get("trend_alignment")
            })
        except Exception as err:
            print("⚠️  JSON parse failed:", err)
            row.update({"decision": "NO_TRADE", "reason": "PARSE_ERROR"})

    # 4a. Persist locally
    save_signal(row)

    # 4b. Append to Google Sheet
    creds  = Credentials.from_service_account_file(
        KEY_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    sheet1 = gspread.authorize(creds).open_by_key(SHEET_ID).sheet1
    if sheet1.row_count == 0 or not sheet1.row_values(1):
        sheet1.append_row(FIELDNAMES, value_input_option="USER_ENTERED")
    sheet1.append_row([row[c] for c in FIELDNAMES],
                       value_input_option="USER_ENTERED")

    print("Logged to CSV and Google Sheets.")
    return row


# ───────────────────────────────────────────────────────────────────────────────
# 2) News-driven volatility → Grok version
# ───────────────────────────────────────────────────────────────────────────────

def _fetch_summaries(ticker: str) -> List[str]:
    """Return up to 25 latest news *summaries* for *ticker*."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        #"topics": TOPICS,
        "time_from": "20250615T0000",
        "sort": "LATEST",
        "limit": 25,
        "apikey": AV_KEY,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
    data = resp.json()
    return [a.get("summary", "") for a in data.get("feed", []) if a.get("summary")]


def find_volatility_grok(
    ticker: str,
    volatility_threshold: float = 0.45,
    *,
    risk: float | None = None,
    verbose: bool = False,
) -> tuple[str, float]:
    """Return ("NO_TRADE (Volatility)", score) or ("TRADE_OK", score)."""
    threshold = risk if risk is not None else volatility_threshold

    # 1) Fetch news
    try:
        summaries = _fetch_summaries(ticker)
    except Exception:
        if verbose:
            print("[VOL] Fetch error → NO_TRADE")
        return "NO_TRADE (Volatility)", 1.0

    if verbose:
        print(f"[VOL] {ticker}: {len(summaries)} summaries fetched")
    if not summaries:
        return "TRADE_OK", 0.0

    # 2) Build prompt
    
    prompt = (""" 
        SYSTEM PROMPT
        You are a financial news analyst who assesses short-term stock volatility based on provided news summaries.

        """
        f"Evaluate the likelihood (probability) that the stock {ticker} will experience significant price volatility (±15 percent or more) within the next 7 calendar days."
        """
        RULES

        If a dividend payment or financial earnings report is scheduled within the next 7 calendar days, immediately assign maximum volatility (1.0).
        
        Otherwise, estimate volatility based on the provided news.

        RESPONSE FORMAT
        Provide only a single decimal number from 0.0 (low volatility) to 1.0 (high volatility).

        EXAMPLES

        0.15

        0.80

        1.0 (Mandatory if dividends or earnings reports are within 7 days)

       
        """
        
        f"News summaries:\n" + "\n\n".join(f"• {s}" for s in summaries) + "\n\nVolatility score:"

        "Volatility score:"

    )

    # 3) Query Grok
    resp = grok_client.chat.completions.create(
        model="grok-4-0709",
        seed=69,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        score = float(raw)
    except ValueError:
        raise ValueError(f"Unexpected Grok reply: {raw!r}")

    if verbose:
        print(f"[VOL] {ticker} → score={score:.3f}, threshold={threshold}")

    # 4) Decide
    if score >= threshold:
        return "NO_TRADE (Volatility)", score
    return "TRADE_OK", score