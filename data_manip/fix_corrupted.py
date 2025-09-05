#!/usr/bin/env python
# build_train_jsonl.py
# ------------------------------------------------------------
# Combine candle context, signals, and recommendations into   #
# a single train.jsonl file suitable for OpenAI fine-tuning.   #
# ------------------------------------------------------------

from pathlib import Path, PurePath
import glob, json
from datetime import timedelta

import pandas as pd

# --------------- config ----------------------------------------------------
DATA_CANDLES_DIR      = "data/candles"            # <SYMBOL>_60min_YYYY-MM-DD.parquet
DATA_SIGNALS_DIR      = "data/signals"            # signals_YYYY-MM-DD.parquet
DATA_RECOMMEND_DIR    = "data/recommendations"    # recs_*.parquet  (name pattern flex)
OUT_JSONL             = "train.jsonl"

BAR_INTERVAL          = timedelta(hours=1)   # 60-min bars
N_PRE_BARS            = 70                   # bars before signal
LOOKBACK_DAYS         = 2                   # search back this many days if needed

# (shortened here – keep your full prompt)
COACH_SYS_MSG = "You are a prompt-engineering and trading coach. …"

# --------------- helpers ---------------------------------------------------
def make_trade_id(symbol: str, ts) -> str:
    """Return SYMBOL-YYYY-MM-DDTHH:MM:SS.ffffff (UTC-naïve)."""
    iso = (
        pd.to_datetime(str(ts).strip(), errors="raise")
          .tz_localize(None)                     # drop tz if present
          .isoformat(timespec="microseconds")
    )
    return f"{str(symbol).strip().upper()}-{iso}"

def load_recommendations(recomm_dir: str = DATA_RECOMMEND_DIR) -> dict:
    """Read all rec parquet files → dict{trade_id: {...}}"""
    rec_dict = {}
    files    = glob.glob(f"{recomm_dir}/*.parquet")

    if not files:
        print(f"⚠️  No recommendation parquet files found in {recomm_dir}")
        return rec_dict

    for fp in files:
        df = pd.read_parquet(fp)

        # unify trade_id
        if "trade_id" not in df.columns and {"symbol", "timestamp"}.issubset(df.columns):
            df["trade_id"] = df.apply(
                lambda r: make_trade_id(r.symbol, r.timestamp), axis=1
            )
        elif "trade_id" not in df.columns:
            raise ValueError(
                f"{Path(fp).name} lacks 'trade_id' or ('symbol','timestamp')."
            )

        for r in df.itertuples(index=False):
            rec_dict[r.trade_id] = {
                "prompt_improvements": (
                    r.prompt_improvements
                    if isinstance(r.prompt_improvements, list)
                    else json.loads(r.prompt_improvements)
                    if pd.notna(r.prompt_improvements) else []
                ),
                "reevaluation_attempts": (
                    r.reevaluation_attempts
                    if isinstance(r.reevaluation_attempts, list)
                    else [int(r.reevaluation_attempts)]
                    if pd.notna(r.reevaluation_attempts) else []
                ),
                "final_result": (
                    r.final_result
                    if isinstance(r.final_result, list)
                    else [str(r.final_result)]
                    if pd.notna(r.final_result) else []
                ),
            }

    print(f"✅  Loaded {len(rec_dict):,} recommendation records")
    return rec_dict

# -- your existing candle helper (unchanged) -----------------
CANDLE_ROOT = Path(DATA_CANDLES_DIR)
def load_last_bars(symbol: str,
                   base_day_str: str,
                   ts: pd.Timestamp,
                   n: int = N_PRE_BARS,
                   lookback_days: int = LOOKBACK_DAYS):
    """Return [{o,h,l,c,v}, …] of the last *n* bars before *ts*."""
    collected = pd.DataFrame()
    day_dt    = pd.to_datetime(base_day_str)

    for d in range(lookback_days + 1):
        day_str = (day_dt - timedelta(days=d)).strftime("%Y-%m-%d")
        fp      = CANDLE_ROOT / f"{symbol}_60min_{day_str}.parquet"
        if not fp.exists():
            continue

        df_day = pd.read_parquet(fp)

        # normalise timestamp
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

    if collected.empty():
        return None

    rows = collected.sort_values("timestamp").tail(n)
    return [{"o": r.o, "h": r.h, "l": r.l, "c": r.c, "v": r.v} for r in rows.itertuples()]

# --------------- main ------------------------------------------------------
def main() -> None:
    RECOMM_LOOKUP = load_recommendations()

    out           = Path(OUT_JSONL)
    lines_written = 0

    signal_files = glob.glob(f"{DATA_SIGNALS_DIR}/signals_*.parquet")
    print("Found", len(signal_files), "signals file(s)\n")

    with out.open("w", encoding="utf-8") as fout:
        for sig_fp in signal_files:
            day_str = PurePath(sig_fp).stem.split("_")[1]  # YYYY-MM-DD
            print("Processing", PurePath(sig_fp).name)
            signals = pd.read_parquet(sig_fp).reset_index()   # ensure timestamp column

            for _, row in signals.iterrows():
                try:
                    trade_id = make_trade_id(row["symbol"], row["timestamp"])
                except Exception as e:
                    print(f"⚠️  Skipping row – bad timestamp: {row['symbol']=}, {row['timestamp']=}")
                    continue

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
                        "confidence":  row["confidence"],
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
                     "reevaluation_attempts": [],
                     "final_result": []}
                )

                fout.write(json.dumps({
                    "prompt": (f"<|system|>{COACH_SYS_MSG}"
                               f"<|end|><|user|>{json.dumps(prompt)}<|end|>"),
                    "completion": json.dumps(completion)
                }) + "\n")

                lines_written += 1
                if lines_written % 500 == 0:
                    print("  …", lines_written, "lines so far")

    print(f"\n✅  Wrote {lines_written:,} lines to {out.name}")

# --------------- run guard -------------------------------------------------
if __name__ == "__main__":
    main()
