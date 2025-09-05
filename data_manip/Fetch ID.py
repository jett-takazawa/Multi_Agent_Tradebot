# parquet_to_jsonl.py  –  build JSONL for coach fine-tune
# -------------------------------------------------------
# ✏️  EDIT THESE THREE PATHS ONLY
DATA_CANDLES_DIR = "data/candles"            # <SYMBOL>_60min_YYYY-MM-DD.parquet
DATA_SIGNALS_DIR = "data/signals"                    # signals_YYYY-MM-DD.parquet
OUT_JSONL        = "train.jsonl"

# -------- imports & constants --------
import pandas as pd, json, glob
from pathlib import Path, PurePath
from datetime import timedelta

BAR_INTERVAL   = 15  # 60-min bars
N_PRE_BARS     = 1                   # bars before signal
LOOKBACK_DAYS  = 20                    # search this many days back if needed

COACH_SYS_MSG = (
    "You are a prompt-engineering coach. "
    "Input JSON: symbol, trade_id, volatility_score, pre_trade_candles, "
    "original_outcome.  Respond with prompt_improvements, final_decision, reason."
)

CANDLE_ROOT = Path(DATA_CANDLES_DIR)

# -------- helper --------
def load_last_bars(symbol: str,
                   base_day_str: str,
                   ts: pd.Timestamp,
                   n: int = N_PRE_BARS,
                   lookback_days: int = LOOKBACK_DAYS):
    """
    Return list[dict(o,h,l,c,v)] of last n bars before ts.
    Handles candle files that store epoch seconds in 't' or normal datetimes.
    Returns None if no bars found.
    """
    collected = pd.DataFrame()
    day_dt    = pd.to_datetime(base_day_str)

    for d in range(lookback_days + 1):
        day_str = (day_dt - timedelta(days=d)).strftime("%Y-%m-%d")
        fp      = CANDLE_ROOT / f"{symbol}_15min_{day_str}.parquet"
        if not fp.exists():
            continue

        df_day = pd.read_parquet(fp)

        # normalise timestamp column
        if df_day.index.name and df_day.index.name.lower().startswith("t"):
            df_day = df_day.reset_index()

        if "t" in df_day.columns and "timestamp" not in df_day.columns:
            df_day = df_day.rename(columns={"t": "timestamp"})
            df_day["timestamp"] = pd.to_datetime(df_day["timestamp"], unit="s")

        for col in df_day.columns:
            if col.lower() == "timestamp" and col != "timestamp":
                df_day = df_day.rename(columns={col: "timestamp"})
                break

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
    return [
        {"o": r.o, "h": r.h, "l": r.l, "c": r.c, "v": r.v}
        for r in rows.itertuples()
    ]

# -------- main --------
def main():
    out = Path(OUT_JSONL)
    lines_written = 0

    signal_files = glob.glob(f"{DATA_SIGNALS_DIR}/signals_*.parquet")
    print("Found", len(signal_files), "signals file(s)\n")

    with out.open("w", encoding="utf-8") as fout:
        for sig_fp in signal_files:
            day_str = PurePath(sig_fp).stem.split("_")[1]  # YYYY-MM-DD
            #print("Processing", PurePath(sig_fp).name)
            signals = pd.read_parquet(sig_fp)

            for r in signals.itertuples():
                bars = load_last_bars(r.symbol, day_str, r.timestamp)
                if bars is None:
                    continue     # skip if no candle data

                prompt = {
                    "symbol": r.symbol,
                    "trade_id": f"{r.symbol}-{r.timestamp}",

                    "bars_since_entry": 0,
                    "volatility_score": r.News_Volatility,
                    "pre_trade_candles": bars,
                    "original_prompt": "detect supply-demand zones",
                    "original_outcome": {
                        "entry":       r.entry,
                        "stop":        r.stop,
                        "exit":        r.exit,
                        "Odds_Score":  r.Odds_Score,
                        "trend":       r.trend,
                        "odds_enhancers": {
                            "strength":        r.strength,
                            "time":            r.time,
                            "freshness":       r.freshness,
                            "trend_alignment": r.trend_alignment
                            
                        }
                    },
                    "final_decision": r.decision,
                    "reason": r.reason
                }

                completion = {              #LOOK FOR THIS GUY YOU ARE NEXT BABY
                    "prompt_improvements": [],
                    "reevaluation_attempts": [],
                    "final_result": []
                    
                }

                fout.write(json.dumps({
                    "prompt":     f"<|system|>{COACH_SYS_MSG}"
                                  f"<|end|><|user|>{json.dumps(prompt)}<|end|>",
                    "completion": json.dumps(completion)
                }) + "\n")

                lines_written += 1
                print(prompt["trade_id"])
                #if lines_written % 500 == 0:
                #    print("  …", lines_written, "lines so far")


    print(f"\n✅  Wrote {lines_written:,} lines to {out.name}")

# -------- run-guard --------
if __name__ == "__main__":
    main()
