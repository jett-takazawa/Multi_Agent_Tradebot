## Jett’s Tradebot — Multi-Agentic Trading System

A production-ready system to **propose, challenge, and approve** trades using engineered time-series features, EV/VaR-aware policies, and supervised-fine-tuned LLM agents. The design follows an **Agentic Flow** pattern (Signal → Risk → Executive) so every decision is **explainable** and **auditable**. Implements the **Local + LLM** approach: heavy analytics run locally; LLMs provide reasoning and narrative, never numeric truth.

---

## **T**L;DR

- **Goal:** Generate **EV-gated, risk-aware, auditable** trade decisions with reproducible artifacts.  
- **Stack:** Python • Pandas • PyArrow/Parquet • DuckDB • (optional) Google Sheets via gspread • Vertex AI Gemini (SFT/RFT-ready) • n8n hooks.  
- **Pattern:** **Local + LLM** — features and analytics in Parquet/DuckDB; agents translate context → proposal → challenge → approval with rationale.

---

## Features

- **Agentic flow:** Signal → Risk → Executive with hard/soft policy gates (ATR stops, sizing, exposure caps, regime filters).  
- **Reproducibility:** Idempotent ETL, versioned prompts/configs, Parquet snapshots, JSONL decision logs.  
- **Walk-forward validation:** Time-split CV, leakage guards, override analytics (how often Risk/Executive veto Signal).  
- **Ops-ready artifacts:** CSV/Sheets summaries, rationale excerpts, and audit trails for post-mortems.  
- **Extensible adapters:** Pluggable data sources, features, and (future) broker connectors.

---

## Architecture

```
Market Data (e.g., OHLCV)
        │
        ▼
Idempotent ETL ──► Parquet/DuckDB (Feature Store)
        │
        ▼
Feature Engineering (ATR, gaps, regimes, liquidity)
        │
        ▼
┌─────────────┐      ┌────────────┐      ┌────────────────┐
│ Signal Agent│ ───► │ Risk Agent │ ───► │ Executive Agent│
└─────────────┘      └────────────┘      └────────────────┘
      │                    │                      │
      ▼                    ▼                      ▼
tradelogtosheets.csv    recommendations.json    final_decisions.jsonl ──► reports/* (CSV/Sheets)
```

**Local + LLM:** Numeric outputs are computed locally (feature store / analytics). LLMs reason over **structured context** and produce human-readable rationales.


---

## Data & Features

- **Primary inputs:** OHLCV bars (provider-agnostic).  
- **Versioning:** Each feature release is stored with the Parquet snapshot to guarantee backtest replayability.
- **Outputs** All outputs from each trained agent is stored in a json format for future fine tuning.

---

## Model Stack

- **Signal Agent:** Generates candidate entries/sides/structures from feature context.  
- **Risk Agent (SFT):** Applies gates (ATR-scaled stops, EV thresholds, exposure/position limits, regime disqualifiers).  
- **Executive Agent (SFT, RFT-ready):** Final **APPROVE/MODIFY/REJECT** with concise rationale; tracks drift and override stats.

**Training data:** Curated JSONL with time-split folds; prompts/system messages are versioned and hash-logged.

---

## Decision Logging & Governance

All agent interactions are recorded with timestamps, model IDs, prompt hashes, and config refs.

```json
{
  "ts": "2025-09-03T14:22:31Z",
  "symbol": "AAPL",
  "agent": "executive",
  "decision": "APPROVE",
  "position": {"side":"LONG","type":"shares","qty":100,"stop":194.2,"target":203.5},
  "rationale": "EV ≥ 0.07 with ATR-scaled risk; reject if regime flips to HiVol-Q4.",
  "prompt_hash": "sha256:…",
  "config_ref": "strategy.default.yaml",
  "input_refs": ["risk_challenge:12345","signal_proposal:67890"]
}
```

Enables **forensics** (what was known/assumed) and **audit readiness** across runs.


---

## Operations & Artifacts

- `data/decisions/*.jsonl` — full decision trail (proposals, challenges, final calls)  
- `reports/*` — daily/periodic summaries (CSV; optional Google Sheets)  
- `backtests/outputs/*` — walk-forward metrics, override matrices, regime summaries

---

## Extensibility

- **Data providers:** Add an ingestor; adhere to the feature store schema.  
- **Brokers (future):** Thin adapter to consume final decisions and post executions (paper/live).  
- **Features:** Drop-in modules; snapshots ensure reproducible replays.  
- **Policies/Prompts:** Centralized in `configs/` and `utils/prompts.py` for variant strategies.

---

## Trades & Snapshots

Log real trades here. Paste screenshots/links for fast executive review.

### Trade Log 

Time stamp	        Symbol	Trend	Pattern	      Entry	Stop	Exit	Odds_Score	Strength	Time	Freshness	Trend_Alignment	    Confidence _NEWS	Risk Ratio
2025-07-17 13:10:00	ASTS	uptrend	DBR	      51.93	51.74	52.65	0.8888888889	Strong	        Strong	Fresh	         With                0.850	         3.8 / 1

<img width="1686" height="915" alt="image" src="https://github.com/user-attachments/assets/79b0ed00-d83c-4e1e-a3ed-05538ab650f1" />

Time stamp	        Symbol	Trend	Pattern	      Entry	Stop	Exit	Odds_Score	Strength	Time	Freshness	Trend_Alignment	     Confidence _NEWS	Risk Ratio
2025-07-15 2:50:21	TREX	uptrend	RBR	        60.27	59.68	64.05	0.944	        Strong	        Good	 Fresh	          With	               0	            6.4 / 1
<img width="1675" height="833" alt="image" src="https://github.com/user-attachments/assets/80878d63-c94b-41a0-8dcd-2f8fb1f7df4a" />

Time stamp	        Symbol	Trend	   Patter      Entry	Stop	Exit	Odds_Score	Strength	Time	Freshness	Trend_Alignment	     Confidence _NEWS	Risk Ratio
2025-07-15 6:20:50	BLDR	downtrend  RBD	       132.6	133.35	129.81	0.8888888889	Strong	        Strong	Fresh	        With	                0.6	             3.7 / 1
<img width="1692" height="826" alt="image" src="https://github.com/user-attachments/assets/eb589d24-b53c-46c0-95d1-7faec53fb863" />


---

## Security, Compliance & Ethics

- Use read-only data keys and segregated credentials.  
- Log decisions, not PII.  
- For **research/education**; **not financial advice**. Verify local regulations before any live trading.

---

## Known Limitations

- Provider constraints (rate limits, corp actions) may affect data fidelity.  
- Slippage/liquidity are modeled; calibrate to venue/instrument.  
- LLM variability persists; gating reduces but does not eliminate variance.

---

## Roadmap

- [ ] RFT-aligned Executive (PnL-aware rewards)  
- [ ] Paper/live broker adapters with event-sourced fills
- [ ] Automated Trade Managment 
- [ ] Regime-aware prompt/persona shifts  
- [ ] Intraday microstructure features  
- [ ] One-click PDF/Slides “Strategy Cards” from logs

---

## Contributing

- Tests: `pytest -q`  
- Style: `ruff check . && ruff format .`  
- PRs welcome—include minimal reproducible examples and environment details.

---

## License

MIT (or update to your preferred license).
