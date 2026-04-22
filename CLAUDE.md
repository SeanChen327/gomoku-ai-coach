# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

An interactive Gomoku (Five-in-a-Row, 15×15) web app where users play against a computer opponent and receive real-time strategic coaching via an AI chat interface. The coaching pipeline combines rule-based board analysis, RAG (Retrieval-Augmented Generation) from a Pinecone vector store seeded with RenjuNet strategies, and Google Gemini LLM responses. The project also supports server-side AI-vs-AI battle simulation and includes an AI governance/assurance layer.

## Tech Stack

- **Backend:** FastAPI (Python), with Render as primary cloud host and Vercel as alternative
- **LLM / Embeddings:** Google Gemini (`gemini-2.5-flash` for chat, `gemini-embedding-001` for 768-dim vectors)
- **Vector DB:** Pinecone (index: `tictactoe-rag`, dimension: 768)
- **Database:** PostgreSQL (Render production) / SQLite (local dev), via SQLAlchemy ORM
- **Auth:** OAuth2 + JWT (python-jose), password hashing via passlib/bcrypt
- **Frontend:** Vanilla HTML/CSS/JS (single `index.html`), no build step
- **Config:** `python-dotenv` — secrets loaded from `.env`
- **Testing:** pytest, httpx, Playwright (E2E), Locust (load testing)

## Environment Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # dev/test deps only
cp .env.example .env   # then fill in keys
```

Required `.env` keys (app raises `ValueError` on startup if missing):
```
GEMINI_API_KEY=...
PINECONE_API_KEY=...
```

Optional/production `.env` keys:
```
DATABASE_URL=postgresql://...   # defaults to SQLite locally
SECRET_KEY=...                  # JWT signing secret
CRON_SECRET=...                 # guards /api/internal/* endpoints
MOCK_AI=true                    # set in CI to bypass Gemini calls
```

## Running the App

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend and API are both served from the same process: `/` → `index.html`, `/api/*` → FastAPI.

## Pinecone Knowledge Base

The original `setup_db.py` has been replaced by a two-step RenjuNet pipeline:

```bash
# Step 1: Scrape strategy content from RenjuNet
python scrape_renjunet.py          # outputs renjunet_knowledge.json

# Step 2: Embed and upsert into Pinecone
python ingest_renjunet.py          # reads renjunet_knowledge.json → Pinecone

# To wipe the index before re-ingesting
python clear_pinecone.py
```

## Architecture

### Request Flow

```
User asks coach question
  → POST /api/chat  {message, board: [225-element list], last_evaluation: {...}}
  → [AI Governance] detect_adversarial_input()   # prompt injection check
  → analyze_board()                               # rule-based: detects threats/forks
  → retrieve_from_pinecone()                      # embeds query → top-2 Pinecone matches (score > 0.5)
  → build system prompt (role + board state + tactical analysis + RAG context)
  → Gemini generate_content()
  → [AI Governance] validate_output_safety()      # hallucination + policy check
  → [AI Governance] track_telemetry()             # latency, tokens, cost
  → return {"reply": "..."}
```

### Key Files

| File | Role |
|---|---|
| `main.py` | FastAPI app: `ChatRequest` model, `analyze_board()`, `retrieve_from_pinecone()`, `/api/chat`, `/api/health`, auth & DB endpoints |
| `ai_governance.py` | `GomokuAIGovernance`: input guardrails, output validation, telemetry, consistency evaluation, HITL flagging |
| `ai_battle_engine.py` | `GomokuSimulator`: server-side headless AI-vs-AI 15×15 match simulation (heuristic scoring, no LLM) |
| `scrape_renjunet.py` | Scrapes RenjuNet strategy articles → `renjunet_knowledge.json` |
| `ingest_renjunet.py` | Embeds JSON chunks via Gemini → upserts vectors into Pinecone |
| `clear_pinecone.py` | Admin utility: deletes all vectors from the Pinecone index |
| `test_engine.py` | Quick smoke test for `GomokuSimulator` |
| `index.html` | All frontend: 15×15 Gomoku board UI, heuristic AI engine (JS), difficulty selector, chat widget, CSV export |
| `vercel.json` | Routing: `/api/*` → `main.py`, everything else → `index.html` (Vercel deployment) |
| `render.yaml` | IaC for Render: Python web service + managed PostgreSQL |
| `GOVERNANCE.md` | Decision log, PR protocol, and feature audit trail |

### Board Representation

225-element list, 15×15 grid (row-major):
```
index = row * 15 + col   (row, col both 0-based)
```
Algebraic coordinates: columns A–O, rows 1–15 (e.g., index 0 = A1, index 224 = O15).
`"X"` = player (Black), `"O"` = computer (White), `""` = empty.

### AI Difficulty

All difficulty levels use a **heuristic evaluation engine** (JS, client-side). Minimax is not used — the 225-cell state space makes it infeasible.

- **Easy:** `getEasyMove()` — epsilon-greedy: always blocks/wins on critical threats (score ≥ 10000), otherwise 60% random / 40% optimal
- **Medium/Hard:** Full heuristic scoring with open-three/four pattern detection

### AI Governance Layer (`ai_governance.py`)

Implements the AI Periodic Table concepts:

| Method | Category | Purpose |
|---|---|---|
| `detect_adversarial_input()` | Guardrails / Red Teaming | Regex blocks prompt injection |
| `validate_output_safety()` | Guardrails | Word-count policy + coordinate hallucination check against board |
| `track_telemetry()` | Metrics | Latency, estimated tokens, estimated cost |
| `evaluate_response_consistency()` | Evaluation | Compares LLM win-rate claim vs deterministic frontend payload |
| `requires_human_oversight()` | Human-in-the-Loop | Flags low-quality responses (score < 0.7) via `logger.critical` |

### Deployment

**Render (primary):** `render.yaml` provisions a Python web service + free PostgreSQL instance. The FastAPI app serves `index.html` via `FileResponse` at `/`.

**Vercel (alternative):** `vercel.json` routes `/api/*` to the Python backend and serves `index.html` as static. No build pipeline required.

## Tests

Automated tests live in `tests/`:

| File | Type | Notes |
|---|---|---|
| `tests/test_ai_engine.py` | Unit | GomokuSimulator correctness, coordinate math, heuristic weights |
| `tests/test_ai_governance.py` | Unit | Guardrails, hallucination detection, telemetry accuracy |
| `tests/test_api_integration.py` | Integration | FastAPI endpoints via httpx |
| `tests/test_e2e_frontend.py` | E2E | Playwright browser automation |
| `tests/test_llm_outputs.py` | LLM-as-judge | Skipped in standard CI (`@pytest.mark.skipif` on `MOCK_AI`) |
| `tests/load_testing/locustfile.py` | Load | Locust simulated user swarm |
| `test_engine.py` | Smoke | Quick GomokuSimulator sanity check (run directly) |

```bash
pytest tests/                          # run all automated tests
pytest tests/ -k "not llm"            # skip LLM-as-judge tests
MOCK_AI=true pytest tests/            # CI mode — bypasses Gemini SDK
```

## PR & Review Protocol

See `GOVERNANCE.md` for the full decision log. All features use `feature/<name>` branches. PRs require:
- **Primary Peer Reviewer:** Ruby (@xxandy-what)
- **Technical Consultant:** Sean (@SeanChen327)
