#!/usr/bin/env python3
import os, csv, json, io, time, datetime as dt, requests, pandas as pd
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
import requests
from openai import OpenAI
import pyarrow as pa, pyarrow.parquet as pq
from typing import List
from data_manip.Data_Collect import save_signal, save_reccs, save_exec
# import gspread # Already present
from google.oauth2.service_account import Credentials
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from google.oauth2 import service_account # 👈 Add this import




# ── static config ──────────────────────────────────────────
RESOLUTION_MIN   = 15         # 0=daily, 5=5-min, etc.
BARS_PER_PROMPT  = 280
PROJECT_ID = "id"  # 
LOCATION = "us-central1"            #
CREDENTIALS_FILE = "CRed.json" # 
creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)

grok_key = ("xai")
if not grok_key:
    raise RuntimeError("GROK_API_KEY is not set or is empty")

grok_client = OpenAI(
    api_key=grok_key,
    base_url="https://api.x.ai/v1",  # your xAI region
    timeout=4500,                     # ← raise from 600 s to 3600 s
    max_retries=2                     # optional: retry transient errors
)

CSV_FILE         = "tradelogtosheets.csv"
AV_KEY           = ""
OA_KEY           = ""
KEY_FILE         = "Credentials.json"
SHEET_ID         = ""
DATA_DIR_CANDLES  = "data/candles" 
DATA_DICT = {}
FIELDNAMES = [
    "timestamp","symbol","trend","pattern",
    "entry","stop","exit","Odds_Score",
    "strength","time","freshness","trend_alignment", "News_Volatility", "Executive"
]


SIGNAL_ENDPOINT_NAME = "projects/adept-tangent-467005-q0/locations/us-central1/endpoints/(endpoint)"
RISK_ENDPOINT_NAME   = "projects/adept-tangent-467005-q0/locations/us-central1/endpoints/(endpoint"
EXEC_ENDPOINT_NAME   = "projects/adept-tangent-467005-q0/locations/us-central1/endpoints/(endpoint)"




SYSTEM_PROMPT = """
You are an advanced Supply-and-Demand Pattern Detector for equities, precisely identifying trade zones using strict breakout and odds enhancer criteria. ALWAYS FIND A ZONE.

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



REEVALUATED_SYSTEM_PROMPT = """

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

Never explain yourself. Only respond with the JSON trade object above.""".strip()




def call_vertex_model(endpoint_name: str, system_prompt: str, user_prompt: str) -> str:
    """
    Calls a tuned Gemini endpoint with system and user prompts.
    Expect endpoint_name like: projects/{PROJECT}/locations/{LOC}/endpoints/{ENDPOINT_ID}
    """
    # (Optional) minimal sanity check
    if "/endpoints/" not in endpoint_name:
        raise ValueError(f"Expected an endpoint name, got: {endpoint_name}")

    model = GenerativeModel(
        endpoint_name,
        system_instruction=[system_prompt]
    )
    response = model.generate_content([user_prompt])
    try:
        return response.text
    except Exception as e:
        print(f"⚠️ Error getting model response: {e}")
        return ""
    
    
# ── helper funcs ───────────────────────────────────────────
def fetch_recent_ohlcv(symbol, resolution_min, n_bars) -> pd.DataFrame:
    if resolution_min:
        interval = f"{resolution_min}min"
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}"
            f"&interval={interval}"
            f"&outputsize=compact"
            f"&datatype=csv"
            f"&apikey={AV_KEY}"
        )
    else:
        interval = "daily"  # ← define for filename
        url = (
            "https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={symbol}"
            f"&outputsize=compact"
            f"&datatype=csv"
            f"&apikey={AV_KEY}"
        )

    csv_text = requests.get(url, timeout=15).text
    if "Thank you for using" in csv_text or csv_text.startswith("{"):
        raise RuntimeError("Alpha Vantage throttle/error:\n" + csv_text[:120])

    df = (
        pd.read_csv(io.StringIO(csv_text))
          .iloc[::-1]
          .tail(n_bars)
          .rename(columns={"timestamp":"t","open":"o","high":"h","low":"l","close":"c","volume":"v"})
          .reset_index(drop=True)
    )
    # Newer pandas: use .view("int64") instead of astype(int)
    df["t"] = pd.to_datetime(df["t"]).view("int64") // 10**9

    os.makedirs(DATA_DIR_CANDLES, exist_ok=True)
    today = dt.date.today().isoformat()
    path = f"{DATA_DIR_CANDLES}/{symbol}_{interval}_{today}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
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

SYSTEM_PROMPT_RISK = """
    You are a prompt-engineering and trading coach. Your job is to analyze a given trade
and return a engineered prompt to advise the trade maker given the candlestick data.
Try to find an alternative zone or reccomendation trade. Keep it short, concise and deliberate.
***If final_result = "trade" or "skip", prompt_improvements = "null"***

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



# PROMPT FOR EXECUTIVE BOT: 


SYSTEM_PROMPT_EXEC_DECISION = """
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




def _extract_json_block(s: str) -> str | None:
    if not s:
        return None
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return s[i:j+1]

def exec_de(signal_json: str, candles_df: pd.DataFrame):
    if signal_json.strip().upper() == "NO_DATA":
        return "no_trade"
    user_msg = build_risk_msg(signal_json, candles_df)
    reply = call_vertex_model(
        endpoint_name=EXEC_ENDPOINT_NAME,
        system_prompt=SYSTEM_PROMPT_EXEC_DECISION,
        user_prompt=user_msg
    ) or ""
    print(reply)
    return reply.strip().lower() or "no_trade"








def build_risk_msg(signal_json: str, candles_df: pd.DataFrame) -> str:
    """
    Build the user message containing only:
      • PROPOSED_ZONE: the JSON string from the signal bot
      • RECENT_CANDLES: the last N candles as plain text
    """
    candle_block = "\n".join(row_to_candle(r) for _, r in candles_df.iterrows())
    return (
        "PROPOSED_ZONE:\n" + signal_json +
        "\n\nRECENT_CANDLES:\n" + candle_block
    )

def risk_check(symbol: str,
               signal_json: str,
               candles_df: pd.DataFrame,
               recommendation: str | None = None) -> dict:

    if signal_json.strip().upper() == "NO_DATA":
        return {"prompt_improvements": "", "final_result": "skip"}

    user_msg = build_risk_msg(signal_json, candles_df)
    if recommendation:
        user_msg += f"\n\nPREVIOUS_RECOMMENDATION:\n{recommendation}"

    resp = call_vertex_model(
        endpoint_name=RISK_ENDPOINT_NAME,
        system_prompt=SYSTEM_PROMPT_RISK,
        user_prompt=user_msg
    ) or ""

    risk_resp = {"prompt_improvements": "", "final_result": "skip"}  # defaults
    
    parsed = json.loads(resp)
    if isinstance(parsed, dict):
        risk_resp["prompt_improvements"] = parsed.get("prompt_improvements")
        risk_resp["final_result"] = parsed.get("final_result")


    # Save merged record only if signal_json is valid JSON
    signalinjson = None
    try:
        signalinjson = json.loads(signal_json)
    except json.JSONDecodeError:
        signalinjson = None

    if isinstance(signalinjson, dict):
        signalinjson.update({
            "prompt_improvements":  risk_resp.get("prompt_improvements", ""),
            "final_result":          risk_resp.get("final_result", "skip"),
        })
        try:
            save_reccs(signalinjson)
        except Exception as e:
            print(f"save_reccs failed: {e}")

    return risk_resp




def run_sys(symbol: str, vol_score: float):
    row = {c: None for c in FIELDNAMES}
    row.update({"symbol": symbol, "timestamp": dt.datetime.utcnow().isoformat(), "News_Volatility": vol_score})

    # 1) candles
    try:
        candles = fetch_recent_ohlcv(symbol, RESOLUTION_MIN, BARS_PER_PROMPT)
    except Exception as e:
        print(f"{symbol}: candle fetch ⚠️ {e}")
        row.update({"decision": "NO_TRADE", "reason": "NO_DATA", "trend": "null"})
        save_signal(row); return

    # 2) signal
    sig_reply = call_vertex_model(
        endpoint_name=SIGNAL_ENDPOINT_NAME,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_msg(candles, symbol)
    ) or ""

    sig_reply_block = _extract_json_block(sig_reply)
    if not sig_reply_block:
        # No JSON → skip safely
        row.update({
            "decision":"SKIP","reason":"NO_JSON_FROM_SIGNAL","trend":"null",
            "pattern":"null","entry":"null","stop":"null","exit":"null",
            "Odds_Score":"null","strength":"null","time":"null",
            "freshness":"null","trend_alignment":"null","Executive":"no_trade"
        })
        save_signal(row); save_exec({"timestamp":row["timestamp"],"symbol":row["symbol"],"Executive":"no_trade"}); return
    sig_reply = sig_reply_block

    # 3) risk
    first_risk = risk_check(symbol, sig_reply, candles)
    decision = first_risk.get("final_result", "no_trade")
    print(sig_reply); print(first_risk)

    # 4) act on decision
    if decision == "reevaluate":
        hint = first_risk.get("prompt_improvements", "")
        full_system_reval = REEVALUATED_SYSTEM_PROMPT + (("\n\n" + hint.strip()) if hint else "")
        user_prompt_reval = build_user_msg(candles, symbol) + (f"\n\n{hint}" if hint else "")

        reevaluated = call_vertex_model(
            endpoint_name=SIGNAL_ENDPOINT_NAME,
            system_prompt=full_system_reval,
            user_prompt=user_prompt_reval
        ) or ""

        re_block = _extract_json_block(reevaluated)
        if not re_block:
            # fall back to SKIP (don’t reference reevaluated later)
            row.update({
                "decision":"SKIP","reason":"NO_JSON_FROM_REEVAL","trend":"null",
                "pattern":"null","entry":"null","stop":"null","exit":"null",
                "Odds_Score":"null","strength":"null","time":"null",
                "freshness":"null","trend_alignment":"null","Executive":"no_trade"
            })
        else:
            reevaluated = re_block
            executivedecision = exec_de(reevaluated, candles)
            zone_obj = json.loads(reevaluated)
            enh = zone_obj.pop("odds_enhancers", {}) or {}
            row.update({
                "decision":"TRADE","reason":"PASS",
                "trend": zone_obj.get("trend","null"),
                "pattern": zone_obj.get("pattern","null"),
                "entry": zone_obj.get("entry","null"),
                "stop":  zone_obj.get("stop","null"),
                "exit":  zone_obj.get("exit","null"),
                "Odds_Score": zone_obj.get("Odds_Score","null"),
                "strength": enh.get("strength","null"),
                "time":     enh.get("time","null"),
                "freshness":enh.get("freshness","null"),
                "trend_alignment": enh.get("trend_alignment","null"),
                "Executive": executivedecision
            })

    elif decision == "pass":
        executivedecision = exec_de(sig_reply, candles)
        zone_obj = json.loads(sig_reply)
        enh = zone_obj.pop("odds_enhancers", {}) or {}
        row.update({
            "decision":"TRADE","reason":"PASS",
            "trend": zone_obj.get("trend","null"),
            "pattern": zone_obj.get("pattern","null"),
            "entry": zone_obj.get("entry","null"),
            "stop":  zone_obj.get("stop","null"),
            "exit":  zone_obj.get("exit","null"),
            "Odds_Score": zone_obj.get("Odds_Score","null"),
            "strength": enh.get("strength","null"),
            "time":     enh.get("time","null"),
            "freshness":enh.get("freshness","null"),
            "trend_alignment": enh.get("trend_alignment","null"),
            "Executive": executivedecision
        })

    elif decision == "no_trade" or decision == "skip":
        row.update({
            "decision":"TRADE","reason":"PASS",
            "trend": zone_obj.get("trend","null"),
            "pattern": zone_obj.get("pattern","null"),
            "entry": zone_obj.get("entry","null"),
            "stop":  zone_obj.get("stop","null"),
            "exit":  zone_obj.get("exit","null"),
            "Odds_Score": zone_obj.get("Odds_Score","null"),
            "strength": enh.get("strength","null"),
            "time":     enh.get("time","null"),
            "freshness":enh.get("freshness","null"),
            "trend_alignment": enh.get("trend_alignment","null"),
            "Executive": "no_trade"
        })
    else:
        # Unknown label → safe default
        row.update({
            "decision":"no_trade","reason":f"UNKNOWN_DECISION:{decision}","trend":"null",
            "pattern":"null","entry":"null","stop":"null","exit":"null",
            "Odds_Score":"null","strength":"null","time":"null",
            "freshness":"null","trend_alignment":"null","Executive":"no_trade"
        })

    # 5) persist (guard against missing keys)
    signalupdate = {
        "timestamp": row.get("timestamp"),
        "symbol": row.get("symbol"),
        "trend": row.get("trend","null"),
        "pattern": row.get("pattern","null"),
        "entry": row.get("entry","null"),
        "stop": row.get("stop","null"),
        "exit": row.get("exit","null"),
        "Odds_Score": row.get("Odds_Score","null"),
        "strength": row.get("strength","null"),
        "time": row.get("time","null"),
        "freshness": row.get("freshness","null"),
        "trend_alignment": row.get("trend_alignment","null"),
        "News_Volatility": row.get("News_Volatility"),
        "decision": row.get("decision"),
        "reason": row.get("reason"),
        "Executive": row.get("Executive","no_trade")

    }
    executiveupdate = {
        "timestamp": row.get("timestamp"),
        "symbol": row.get("symbol"),
        "Executive": row.get("Executive","no_trade")
    }
    save_signal(signalupdate); save_exec(executiveupdate)

    try:
        print(f"{symbol:5} → {row['decision']} ({row['reason']})")
        creds = Credentials.from_service_account_file(KEY_FILE, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        sheet1 = gspread.authorize(creds).open_by_key(SHEET_ID).sheet1
        if not sheet1.get_all_values():
            sheet1.append_row(FIELDNAMES, value_input_option="USER_ENTERED")
        sheet1.append_row([signalupdate.get(c, "") for c in FIELDNAMES], value_input_option="USER_ENTERED")
        print("Logged to Google Sheets.")
    except Exception as e:
        print(f"{symbol:5} → sheet write warn: {e}")




#VOLATILITY news test

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


def find_volatility_grok2(
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
