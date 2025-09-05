#!/usr/bin/env python
# build_train_jsonl.py
# ------------------------------------------------------------
# Combine candle context, signals, and a single recommendations file
# plus an extra-trades file into train.jsonl—only for trades that have recommendations
# or exist in the extra file.
# Output in Gemini SFT JSONL format: top-level systemInstruction+contents.
# ------------------------------------------------------------

from pathlib import Path, PurePath
import glob, json
from datetime import timedelta
import pandas as pd

# ----------------- configuration ---------------------------------
DATA_CANDLES_DIR = "data/candles"
DATA_SIGNALS_DIR = "data/signals"
RECOMM_FILE = "data/Training/recommend_2025-07-24.parquet"
EXTRA_FILE = "data/Training/goodtrades_2025-07-24.parquet"
OUT_JSONL = "RISKAGENTtrain.jsonl"

# 15-minute bars
BAR_INTERVAL = 15
N_PRE_BARS = 280
LOOKBACK_DAYS = 20

COACH_SYS_MSG = """
    You are a prompt-engineering and trading coach. Your job is to analyze a given trade
and return a engineered prompt to advise the trade maker given the candlestick data.
Try to find an alternative zone or reccomendation trade. Keep it short, concise and deliberate.
***If final_result = "trade" or "no_trade", prompt_improvements = "null"***

Respond ONLY with a JSON object:
{
  "prompt_improvements": str,
  "final_result": | "reevaluate" | "pass" | "skip"
}

Your given trade is based upon these:         

****   TRADEMAKER INSTRUCTIONS   ****
    
            You are an advanced Supply-and-Demand Pattern Detector for equities, precisely identifying trade zones using strict breakout and odds enhancer criteria. Do your best to find a zone.

            ***You have also been given advice from a fine, tuned reevaluation bot. Apply the reccomended zone to the best of your ability and trust the provided reevaluation. You must find a trade which works in tangent with the reevaltion bot's prompt, there is potential in this chart.***
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

            ***END***



    INPUTED TRADE DATA IN JSON FORMAT: 
                 {symbol, trade_id, bars_since_entry, volatility_score, pre_trade_candles {o(open), h(high), l(low), c(close), v(volume)}
                 original_outcome{entry, stop, exit, cofidence, trend, odds_enhancers {strength, time, freshness, trend_alignment}, final_decision, reason}
                }


    ONLY ACCEPTABLE OUTPUT 
    "original_outcome.  Respond with prompt_improvements, final_decision, reason."
    JSON object with the following keys:
  {
    "prompt_improvements": "str",
    "final_result":reevaluate, pass, skip"                   
                 """.strip()


# ----------------- helpers ---------------------------------------
def coerce_to_list(val):
    import numbers, collections.abc, pandas as pd
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except:
            return [val]
    if isinstance(val, (numbers.Number, bool)):
        return [val]
    if isinstance(val, collections.abc.Iterable):
        return list(val)
    return [val]


def load_recommendations(recomm_file=RECOMM_FILE) -> dict:
    df = pd.read_parquet(recomm_file)
    if 'trade_id' not in df.columns:
        df['trade_id'] = df.apply(
            lambda r: f"{r.symbol.upper()}-{pd.to_datetime(r.timestamp).tz_localize(None).isoformat(timespec='microseconds')}",
            axis=1
        )
    recs = {}
    for r in df.itertuples(index=False):
        tid = r.trade_id
        pimpr = coerce_to_list(getattr(r, 'prompt_improvements', None))
        final = coerce_to_list(getattr(r, 'final_result', None))
        recs[tid] = {
            'prompt_improvements': pimpr[0] if pimpr else "",
            'final_result': final[0] if final else ""
        }
    print(f"✅  Loaded {len(recs):,} recommendation records from {recomm_file}")
    return recs


def load_last_bars(symbol: str, base_day: str, ts: pd.Timestamp,
                   n: int=N_PRE_BARS, lookback_days: int=LOOKBACK_DAYS):
    import pandas as pd
    collected = pd.DataFrame()
    day0 = pd.to_datetime(base_day)
    for d in range(lookback_days+1):
        day = (day0 - timedelta(days=d)).strftime("%Y-%m-%d")
        fp = Path(DATA_CANDLES_DIR) / f"{symbol}_15min_{day}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if 't' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'t':'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        sub = df[df['timestamp'] < ts].sort_values('timestamp')
        if sub.empty:
            continue
        collected = pd.concat([collected, sub]).tail(n)
        if len(collected) >= n:
            break
    if collected.empty:
        return None
    rows = collected.sort_values('timestamp').tail(n)
    return [{'o': r.o, 'h': r.h, 'l': r.l, 'c': r.c, 'v': r.v} for r in rows.itertuples()]

# ----------------- main -------------------------------------------
def main() -> None:
    recs = load_recommendations()
    seen = set()
    out_path = Path(OUT_JSONL)
    count = 0

    with out_path.open('w', encoding='utf-8') as fout:
        # Process signal files
        sig_files = glob.glob(f"{DATA_SIGNALS_DIR}/signals_????-??-??.parquet")
        print(f"Found {len(sig_files)} signal files")
        for fp in sig_files:
            day = PurePath(fp).stem.split('_')[1]
            df = pd.read_parquet(fp).reset_index()
            for _, row in df.iterrows():
                ts = pd.to_datetime(row['timestamp'])
                tid = f"{row['symbol'].upper()}-{ts.tz_localize(None).isoformat(timespec='microseconds')}"
                if tid in seen or tid not in recs:
                    continue
                seen.add(tid)
                bars = load_last_bars(row['symbol'], day, ts)
                if bars is None:
                    continue
                prompt = {
                    'symbol': row['symbol'],
                    'trade_id': tid,
                    'bars_since_entry': 0,
                    'volatility_score': row['News_Volatility'],
                    'pre_trade_candles': bars,
                    'original_outcome': {
                        'entry': row['entry'],
                        'stop': row['stop'],
                        'exit': row['exit'],
                        'Odds_Score': row['Odds_Score'],
                        'trend': row['trend'],
                        'odds_enhancers': {
                            'strength': row['strength'],
                            'time': row['time'],
                            'freshness': row['freshness'],
                            'trend_alignment': row['trend_alignment']
                        }
                    }
                }
                comp = recs[tid]

                # Build Gemini SFT record
                record = {
                    'systemInstruction': {
                        'role': 'system',
                        'parts': [{'text': COACH_SYS_MSG}]
                    },
                    'contents': [
                        {
                            'role': 'user',
                            'parts': [{'text': json.dumps(prompt, separators=(',', ':'))}]
                        },
                        {
                            'role': 'model',
                            'parts': [{'text': json.dumps(comp, separators=(',', ':'))}]
                        }
                    ]
                }
                fout.write(json.dumps(record, separators=(',', ':')) + '\n')
                count += 1

        # Process extra trades
        extra_df = pd.read_parquet(EXTRA_FILE)
        print(f"Processing extra trades: total {len(extra_df)} records")
        for row in extra_df.itertuples(index=False):
            tid = str(row.trade_id)
            if tid in seen:
                continue
            seen.add(tid)
            ts = pd.to_datetime(row.timestamp, unit='ms')
            day = ts.strftime('%Y-%m-%d')
            bars = load_last_bars(row.symbol, day, ts)
            if bars is None:
                continue
            prompt = {
                'symbol': row.symbol,
                'trade_id': tid,
                'bars_since_entry': 0,
                'volatility_score': row.News_Volatility,
                'pre_trade_candles': bars,
                'original_outcome': {
                    'entry': row.entry,
                    'stop': row.stop,
                    'exit': row.exit,
                    'Odds_Score': row.Odds_Score,
                    'trend': row.trend,
                    'odds_enhancers': {
                        'strength': row.strength,
                        'time': row.time,
                        'freshness': row.freshness,
                        'trend_alignment': row.trend_alignment
                    }
                }
            }
            comp = {'prompt_improvements': 'none', 'final_result': 'pass'}
            record = {
                'systemInstruction': {
                    'role': 'system',
                    'parts': [{'text': COACH_SYS_MSG}]
                },
                'contents': [
                    {
                        'role': 'user',
                        'parts': [{'text': json.dumps(prompt, separators=(',', ':'))}]
                    },
                    {
                        'role': 'model',
                        'parts': [{'text': json.dumps(comp, separators=(',', ':'))}]
                    }
                ]
            }
            fout.write(json.dumps(record, separators=(',', ':')) + '\n')
            count += 1

    print(f"\n✅  Wrote {count:,} lines to {OUT_JSONL}")

if __name__ == '__main__':
    main()
