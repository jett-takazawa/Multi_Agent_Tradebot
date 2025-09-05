#!/usr/bin/env python
# build_train_jsonl.py
# ------------------------------------------------------------
# Combine candle context, signals, a single exec‐decision file,
# plus an extra-trades file into EXECUTIVEAGENTtrain.jsonl.
# ------------------------------------------------------------

from pathlib import Path, PurePath
import glob, json
from datetime import timedelta
import pandas as pd

# ----------------- configuration ---------------------------------
DATA_CANDLES_DIR   = "data/candles"
DATA_SIGNALS_DIR   = "data/signals"
RECOMM_FILE        = "data/Training/execDecision_2025-07-24.parquet"
EXTRA_FILE         = "data/Training/goodtrades_2025-07-24.parquet"
OUT_JSONL          = "EXECUTIVEAGENTtrain.jsonl"

# 15-minute bars
N_PRE_BARS    = 280
LOOKBACK_DAYS = 20


COACH_SYS_MSG = """
You are the Executive Decision Bot. Your sole task is to evaluate a proposed equity trade using the supplied trade data and recent candlestick history, and decide if it is profitable.

Respond with exactly one word, lowercase, and nothing else:
- trade
- no_trade

****   TRADEMAKER INSTRUCTIONS   ****

You are an advanced Supply-and-Demand Pattern Detector for equities, precisely identifying trade zones using strict breakout and odds enhancer criteria. Do your best to find a zone.

You have also been given advice from a fine-tuned reevaluation bot. Apply the recommended zone to the best of your ability and trust the provided reevaluation. You must find a trade which works in tangent with the reevaluation bot’s prompt—there is potential in this chart.

Allowed Output:
- Exactly “trade” if the evaluated setup meets your criteria for a profitable trade.
- Exactly “no_trade” otherwise.

Pattern Criteria:
- Demand Zones (valid only in an uptrend or sideways market): DBR (Drop-Base-Rally), RBR (Rally-Base-Rally)
- Supply Zones (valid only in a downtrend or sideways market): RBD (Rally-Base-Drop), DBD (Drop-Base-Drop)

Zone Identification:
- Analyze backward from the most recent candle, focusing on the first significant breakout within 20 candles.
- Proximal Line (Entry Price): Highest candle body within the identified base.
- Distal Line (Stop-Loss Price):
  - DBR: Lowest wick among leg-in, base, and leg-out candles.
  - RBR: Lowest wick within the base.
  - RBD: Highest wick among leg-in, base, and leg-out candles.
  - DBD: Highest wick within the base and leg-out candle.
- Always prioritize the most recently formed valid zone.

Odds Enhancers (sum points ÷ 9 = Odds_Score):
1. Strength (Zone Departure): Strong (≥2× zone height + breakStructure = +2), Good (≥2× or breakStructure = +1), Poor (0).
2. Time at Base: Strong (1–3 candles = +1), Good (4–6 = +0.5), Poor (>6 = 0).
3. Freshness: Fresh (proximal untouched = +2), Poor (≤50% penetration = +1), Violated (exceeded distal = 0).
4. Trend Alignment: With (+2), Sideways (+1), Counter (0).
5. Risk-Reward Ratio (>3:1 = +1, >6:1 = +2).

Exit Price: Proximal line of the next opposing zone (above entry for demand, below entry for supply).

***END TRADEMAKER INSTRUCTIONS***

INPUT FORMAT (JSON):
{
  "trade": {
    "symbol": str,
    "entry": float,
    "stop": float,
    "exit": float,
    "trend": "uptrend"|"downtrend"|"sideways",
    "Odds_Score": float,
    "odds_enhancers": {
      "strength": "Strong"|"Good"|"Poor",
      "time": "Strong"|"Good"|"Poor",
      "freshness": "Fresh"|"Poor"|"Violated",
      "trend_alignment": "With"|"Sideways"|"Counter"
    }
  },
  "candles": [
    {"o": float, "h": float, "l": float, "c": float, "v": float}, ...
  ]
}

Remember: use the philosophy above to decide if this setup is worth trading.  Then reply **only** with `trade` or `no_trade`.
""".strip()


# ----------------- helpers ---------------------------------------
import numbers, collections.abc

def make_trade_id(symbol: str, ts) -> str:
    dt = pd.to_datetime(ts)
    return f"{symbol.upper()}-{dt.tz_localize(None).isoformat(timespec='microseconds')}"


def coerce_to_list(val):
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
            lambda r: make_trade_id(r.symbol, r.timestamp),
            axis=1
        )
    rec = {}
    for r in df.itertuples(index=False):
        raw = getattr(r, 'exec_output', None) or getattr(r, 'final_result', None)
        lst = coerce_to_list(raw)
        decision = lst[0].lower() if lst else ""
        rec[r.trade_id] = decision
    print(f"✅  Loaded {len(rec):,} recommendation records from {recomm_file}")
    return rec

# candle loader
CANDLE_ROOT = Path(DATA_CANDLES_DIR)

def load_last_bars(symbol: str, base_day: str, ts: pd.Timestamp,
                   n: int=N_PRE_BARS, lookback_days: int=LOOKBACK_DAYS):
    df_all = pd.DataFrame()
    base = pd.to_datetime(base_day)
    for d in range(lookback_days+1):
        day = (base - timedelta(days=d)).strftime("%Y-%m-%d")
        fp = CANDLE_ROOT / f"{symbol}_15min_{day}.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp)
        if 't' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'t':'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df_all = pd.concat([df_all, df[df['timestamp']<ts]])
        if len(df_all) >= n:
            break
    if df_all.empty: return None
    rows = df_all.sort_values('timestamp').tail(n)
    return [{'o':r.o,'h':r.h,'l':r.l,'c':r.c,'v':r.v} for r in rows.itertuples()]

# ----------------- main -------------------------------------------
def main() -> None:
    recs = load_recommendations()
    seen = set()
    out = Path(OUT_JSONL)
    count = 0

    with out.open('w', encoding='utf-8') as fout:
        sig_files = glob.glob(f"{DATA_SIGNALS_DIR}/signals_????-??-??.parquet")
        print(f"Found {len(sig_files)} signal files")
        for fp in sig_files:
            day = PurePath(fp).stem.split('_')[1]
            df = pd.read_parquet(fp).reset_index()
            for _, r in df.iterrows():
                ts = pd.to_datetime(r['timestamp'])
                tid = make_trade_id(r['symbol'], ts)
                if tid in seen or tid not in recs: continue
                seen.add(tid)
                bars = load_last_bars(r['symbol'], day, ts)
                if bars is None: continue

                prompt_payload = json.dumps({
                    'trade': {
                        'symbol': r['symbol'],
                        'entry': r['entry'],
                        'stop': r['stop'],
                        'exit': r['exit'],
                        'trend': r['trend'],
                        'Odds_Score': r['Odds_Score'],
                        'odds_enhancers': {
                            'strength': r['strength'],
                            'time': r['time'],
                            'freshness': r['freshness'],
                            'trend_alignment': r['trend_alignment']
                        }
                    },
                    'candles': bars
                }, separators=(',',':'))

                decision = recs[tid]

                record = {
                    'systemInstruction': {'role':'system','parts':[{'text':COACH_SYS_MSG}]},
                    'contents': [
                        {'role':'user', 'parts':[{'text': prompt_payload}]},
                        {'role':'model','parts':[{'text': decision}]}  
                    ]
                }
                fout.write(json.dumps(record, separators=(',',':')) + '\n')
                count += 1

        extra_df = pd.read_parquet(EXTRA_FILE)
        print(f"Processing extra trades: total {len(extra_df)} records")
        for r in extra_df.itertuples(index=False):
            tid = make_trade_id(r.symbol, r.timestamp)
            if tid in seen: continue
            seen.add(tid)
            ts = pd.to_datetime(r.timestamp, unit='ms')
            day = ts.strftime('%Y-%m-%d')
            bars = load_last_bars(r.symbol, day, ts)
            if bars is None: continue

            prompt_payload = json.dumps({
                'trade': {'symbol': r.symbol, 'entry': r.entry, 'stop': r.stop, 'exit': r.exit,
                          'trend': r.trend, 'Odds_Score': r.Odds_Score,
                          'odds_enhancers': {'strength': r.strength, 'time': r.time,
                                             'freshness': r.freshness, 'trend_alignment': r.trend_alignment}},
                'candles': bars
            }, separators=(',',':'))

            decision = 'trade'

            record = {
                'systemInstruction': {'role':'system','parts':[{'text':COACH_SYS_MSG}]},
                'contents': [
                    {'role':'user', 'parts':[{'text': prompt_payload}]},
                    {'role':'model','parts':[{'text': decision}]}  
                ]
            }
            fout.write(json.dumps(record, separators=(',',':')) + '\n')
            count += 1

    print(f"\n✅  Wrote {count:,} lines to {OUT_JSONL}")

if __name__ == '__main__':
    main()
