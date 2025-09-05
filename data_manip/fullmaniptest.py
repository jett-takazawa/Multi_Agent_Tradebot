# parquet_to_jsonl.py  –  build JSONL for coach fine-tune
# -------------------------------------------------------
#  EDIT THESE THREE PATHS ONLY
DATA_CANDLES_DIR = "data/candles"            # <SYMBOL>_60min_YYYY-MM-DD.parquet
DATA_SIGNALS_DIR = "data/signals"                    # signals_YYYY-MM-DD.parquet
DATA_RECOMMEND_DIR = "data/recommendations" ##THIS IS WHAT I WANT TO ADD TO MY NEW CODE 
OUT_JSONL        = "RISKBOT_TRAIN.jsonl"

# -------- imports & constants --------
import pandas as pd, json, glob
from pathlib import Path, PurePath
from datetime import timedelta

BAR_INTERVAL   = timedelta(hours=1)   # 60-min bars
N_PRE_BARS     = 70                   # bars before signal
LOOKBACK_DAYS  = 2                    # search this many days back if needed

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
                 original_outcome{entry, stop, exit, cofidence, trend, odds_enhancers {strength, time, freshness, trend_alignment}}
                }


    ONLY ACCEPTABLE OUTPUT 
    "original_outcome.  Respond with prompt_improvements, final_decision, reason."
    JSON object with the following keys:
  {
    "prompt_improvements": "str",
    "final_result":reevaluate, pass, skip"                   
                 """.strip()



{"trade_id":"STX-2025-07-13T05:31:42.669012","prompt_improvements":"none","final_result":"skip"}

# ----------------- helpers ---------------------------------------
def make_trade_id(symbol: str, ts) -> str:
    iso = (
        pd.to_datetime(str(ts).strip(), errors="raise")
          .tz_localize(None)
          .isoformat(timespec="microseconds")
    )
    return f"{str(symbol).strip().upper()}-{iso}"

def coerce_to_list(val):
    import numbers, collections.abc, pandas as pd, json
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        s = val.strip()
        if s and s[0] in "[{":
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                pass
        return [s]
    if isinstance(val, (numbers.Number, bool)):
        return [val]
    if isinstance(val, collections.abc.Iterable):
        return list(val)
    return [val]

def load_recommendations(recomm_dir=DATA_RECOMMEND_DIR) -> dict:
    rec_dict = {}
    files = glob.glob(f"{recomm_dir}/*.parquet")
    print("DEBUG rec files ➜", files)

    for fp in files:
        df = pd.read_parquet(fp)
        if df.empty:
            continue

        if "trade_id" not in df.columns:
            if {"symbol", "timestamp"}.issubset(df.columns):
                df["trade_id"] = df.apply(
                    lambda r: make_trade_id(r.symbol, r.timestamp), axis=1
                )
            else:
                print(f"⚠️  {Path(fp).name} missing key columns – skipped")
                continue

        for r in df.itertuples(index=False):
            rec_dict[r.trade_id] = {
                "prompt_improvements":   coerce_to_list(r.prompt_improvements),
                "final_result":          coerce_to_list(r.final_result),
            }

    print(f"✅  Loaded {len(rec_dict):,} recommendation records")
    return rec_dict

# -- candle loader (same as before, abbreviated) -------------------
CANDLE_ROOT = Path(DATA_CANDLES_DIR)
def load_last_bars(symbol: str, base_day_str: str, ts: pd.Timestamp,
                   n: int = N_PRE_BARS, lookback_days: int = LOOKBACK_DAYS):

    collected = pd.DataFrame()
    day_dt    = pd.to_datetime(base_day_str)

    for d in range(lookback_days + 1):
        day_str = (day_dt - timedelta(days=d)).strftime("%Y-%m-%d")
        fp      = CANDLE_ROOT / f"{symbol}_60min_{day_str}.parquet"
        if not fp.exists():
            continue

        df_day = pd.read_parquet(fp)
        if df_day.index.name and df_day.index.name.lower().startswith("t"):
            df_day = df_day.reset_index()

        if "t" in df_day.columns and "timestamp" not in df_day.columns:
            df_day = df_day.rename(columns={"t": "timestamp"})
            df_day["timestamp"] = pd.to_datetime(df_day["timestamp"], unit="s")

        if "timestamp" not in df_day.columns:
            continue

        subset = df_day[df_day["timestamp"] < ts].sort_values("timestamp")
        if subset.empty:
            continue

        collected = pd.concat([collected, subset]).tail(n)
        if len(collected) >= n:
            break

    if collected.empty:
        return None

    rows = collected.sort_values("timestamp").tail(n)
    return [{"o": r.o, "h": r.h, "l": r.l, "c": r.c, "v": r.v} for r in rows.itertuples()]

# ----------------- main ------------------------------------------
def main() -> None:
    RECOMM_LOOKUP   = load_recommendations()
    seen_trade_ids  = set()                     # <-- B) dedup set
    out             = Path(OUT_JSONL)
    lines_written   = 0

    # A) only canonical daily files (no "_clean")
    signal_files = glob.glob(f"{DATA_SIGNALS_DIR}/signals_????-??-??.parquet")
    print("Found", len(signal_files), "signals file(s)\n")

    with out.open("w", encoding="utf-8") as fout:
        for sig_fp in signal_files:
            day_str = PurePath(sig_fp).stem.split("_")[1]     # YYYY-MM-DD
            print("Processing", PurePath(sig_fp).name)
            signals = pd.read_parquet(sig_fp).reset_index()

            for _, row in signals.iterrows():
                try:
                    trade_id = make_trade_id(row["symbol"], row["timestamp"])
                except Exception:
                    continue

                if trade_id in seen_trade_ids:   # B) skip duplicates
                    continue
                seen_trade_ids.add(trade_id)

                bars = load_last_bars(row["symbol"], day_str, row["timestamp"])
                if bars is None:
                    continue

                prompt = {
                    "symbol":            row["symbol"],
                    "trade_id":          trade_id,
                    "bars_since_entry":  0,
                    "volatility_score":  row["News_Volatility"],
                    "pre_trade_candles": bars,
                    "original_prompt":   "detect supply-demand zones",
                    "original_outcome": {
                        "entry":       row["entry"],
                        "stop":        row["stop"],
                        "exit":        row["exit"],
                        "Odds_Score":  row["Odds_Score"],
                        "trend":       row["trend"],
                        "odds_enhancers": {
                            "strength":        row["strength"],
                            "time":            row["time"],
                            "freshness":       row["freshness"],
                            "trend_alignment": row["trend_alignment"],
                        },
                    },
                    "final_decision": row["decision"],
                    "reason":         row["reason"],
                }

                completion = RECOMM_LOOKUP.get(
                    trade_id,
                    {"prompt_improvements": [],
        
                     "final_result": []}
                )

                fout.write(json.dumps({
                    "messages": [
                        { "role": "system", "content": COACH_SYS_MSG },
                        { "role": "user",   "content": json.dumps(prompt, separators=(',',':')) }
                    ],
                    "completion": json.dumps(completion, separators=(',',':')) + "\n"
                    }) + "\n")


                lines_written += 1
                if lines_written % 500 == 0:
                    print("  …", lines_written, "unique lines so far")

    print(f"\n✅  Wrote {lines_written:,} unique lines to {out.name}")

# ----------------- run guard -------------------------------------
if __name__ == "__main__":
    main()