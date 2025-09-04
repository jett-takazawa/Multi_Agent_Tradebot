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
proposals.jsonl     challenges.jsonl        decisions.jsonl ──► reports/* (CSV/Sheets)
```

**Local + LLM:** Numeric outputs are computed locally (feature store / analytics). LLMs reason over **structured context** and produce human-readable rationales.

---

## Repository Structure

```
tradebot/
  agents/
    signal.py
    risk.py
    executive.py
  etl/
    ingest_prices.py
    build_features.py
  backtests/
    runner.py
    metrics.py
  utils/
    io.py
    prompts.py
  configs/
    strategy.default.yaml
    env.example
  data/
    parquet/           # partitioned features
    decisions/         # *.jsonl: proposals/challenges/decisions
  reports/
    daily/             # CSV summaries (optional Sheets sync)
  tests/
    test_*.py
  README.md
```

---

## Data & Features

- **Primary inputs:** OHLCV bars (provider-agnostic).  
- **Feature examples:** ATR bands & percentiles; gap stats; regime flags (trend/vol quartiles); session heuristics (DoW/HoD); liquidity screens (ADV/spread).  
- **Versioning:** Each feature release is stored with the Parquet snapshot to guarantee backtest replayability.

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

## Validation & Backtesting Philosophy

- **Walk-forward evaluation:** Rolling windows; fixed train → shifting test.  
- **Leakage controls:** No look-ahead; strict partitioning for features/labels.  
- **Metrics:** CAGR, Sharpe/Sortino, MaxDD, hit rate, avg R:R, EV distribution, slippage sensitivity.  
- **Agent analytics:** Veto/override rates, common veto reasons, regime-specific performance.

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

### Trade Log (Summary Table)

| Date (UTC) | Symbol | Setup / Thesis | Entry | Stop | Target | R:R | Outcome | EV at Decision | Snapshot | Notes |
|---|---|---|---:|---:|---:|---:|---|---:|---|---|
| 2025-09-03 | AAPL | Gap-reversion + ATR-P20 | 198.10 | 194.20 | 203.50 | 1.4 | +0.9R | +0.07 | *(link/image)* | Exec approved; regime neutral |
| 2025-09-04 | NVDA | Pullback to ATR band | — | — | — | — | — | — | *(add)* | *(add)* |

### Individual Trade Card (Template)

**Symbol / Date:** `TICKER — YYYY-MM-DD`  
**Setup:** *(one-line thesis)*  
**Risk Policy Snapshot:** *(active constraints: ATR14 stop, EV gate, exposure cap, liquidity min)*  
**Executive Rationale (excerpt):**  
> *(Paste 1–3 lines from `decisions.jsonl` “rationale”.)*

**Numbers:**  
- Entry: `…`  Stop: `…`  Target: `…`  Size: `…`  
- EV at decision: `…`  Regime: `…` (e.g., HiVol-Q3)  
- Outcome: `…` (R multiple and %)

**Artifacts:**  
- Screenshot: *(embed image)*  
- Links: *(chart, order ticket, blotter, catalyst)*

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
