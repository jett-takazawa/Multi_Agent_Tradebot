#!/usr/bin/env python
# build_signal_candle_jsonl.py
# ------------------------------------------------------------
# For each trade in data/Training/signal_2025-07-24.parquet, fetch the
# prior N_PRE_BARS fifteen-minute bars and write a JSONL conversation.
# ------------------------------------------------------------

from pathlib import Path
import glob, json
from datetime import timedelta
import pandas as pd

# ----------------- configuration ---------------------------------
DATA_CANDLES_DIR = "data/candles"
SIGNAL_FILE      = "data/Training/signal_2025-07-24.parquet"
OUT_JSONL        = "signal_with_candles.jsonl"

N_PRE_BARS    = 280
LOOKBACK_DAYS = 20

PROMPT_SYS = """
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



# ----------------- candle loader ----------
CANDLE_ROOT = Path(DATA_CANDLES_DIR)
def load_last_bars(symbol: str, ts: pd.Timestamp,
                   n: int = N_PRE_BARS, lookback_days: int = LOOKBACK_DAYS):
    collected = pd.DataFrame()
    base_day = ts.normalize()
    for d in range(lookback_days + 1):
        day_str = (base_day - timedelta(days=d)).strftime("%Y-%m-%d")
        path = CANDLE_ROOT / f"{symbol}_15min_{day_str}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.index.name and df.index.name.lower().startswith("t"):
            df = df.reset_index()
        if "t" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"t": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        if "timestamp" not in df.columns:
            continue
        subset = df[df["timestamp"] < ts].sort_values("timestamp")
        if subset.empty:
            continue
        collected = pd.concat([collected, subset]).tail(n)
        if len(collected) >= n:
            break
    if collected.empty:
        return None
    rows = collected.sort_values("timestamp").tail(n)
    return [{"o": r.o, "h": r.h, "l": r.l, "c": r.c, "v": r.v}
            for r in rows.itertuples()]

# ----------------- main -------------------
def main():
    trades = pd.read_parquet(SIGNAL_FILE)
    out_path = Path(OUT_JSONL)
    count = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for row in trades.itertuples(index=False):
            orig_ts = row.timestamp
            ts      = pd.to_datetime(orig_ts, unit="ms") if isinstance(orig_ts, (int, float)) else pd.to_datetime(orig_ts)
            symbol  = row.symbol.upper()

            bars = load_last_bars(symbol, ts)
            if bars is None:
                continue

            # build payloads
            user_msg   = json.dumps(bars, separators=(",", ":"))
            trade_dict = {
                "timestamp": ts.isoformat(),
                "symbol":    symbol,
                "trend":     row.trend,
                "pattern":   row.pattern,
                "entry":     row.entry,
                "stop":      row.stop,
                "exit":      row.exit,
                "Odds_Score": row.Odds_Score,
                "odds_enhancers": {
                    "strength":        row.strength,
                    "time":            row.time,
                    "freshness":       row.freshness,
                    "trend_alignment": row.trend_alignment
                }
            }
            assist_msg = json.dumps(trade_dict, separators=(",", ":"))

            # --- Gemini-SFT record ---
            record = {
                "systemInstruction": {
                    "role":  "system",
                    "parts": [ { "text": PROMPT_SYS } ]
                },
                "contents": [
                    {
                        "role":  "user",
                        "parts": [ { "text": user_msg } ]
                    },
                    {
                        "role":  "model",
                        "parts": [ { "text": assist_msg } ]
                    }
                ]
            }
            fout.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1

    print(f"✅  Wrote {count:,} lines to {OUT_JSONL}")

if __name__ == "__main__":
    main()