# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

An interactive Tic-Tac-Toe web app where users play against a computer opponent and receive real-time strategic coaching via an AI chat interface. The coaching pipeline combines rule-based board analysis, RAG (Retrieval-Augmented Generation) from a Pinecone vector store, and Google Gemini LLM responses.

## Tech Stack

- **Backend:** FastAPI (Python), served via Vercel Python runtime
- **LLM / Embeddings:** Google Gemini (`gemini-2.5-flash` for chat, `gemini-embedding-001` for 768-dim vectors)
- **Vector DB:** Pinecone (index: `tictactoe-rag`, dimension: 768)
- **Frontend:** Vanilla HTML/CSS/JS (single `index.html`), no build step
- **Config:** `python-dotenv` — secrets loaded from `.env`

## Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

Required `.env` keys (app raises `ValueError` on startup if missing):
```
GEMINI_API_KEY=...
PINECONE_API_KEY=...
```

## Running the App

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend and API are both served from the same process: `/` → `index.html`, `/api/*` → FastAPI.

## Pinecone Knowledge Base

Run once before first use (or after adding new expert tips to `setup_db.py`):

```bash
python setup_db.py
```

This embeds 5 hardcoded expert tips and upserts them into Pinecone as `KB-1` through `KB-5`. Adding new tips requires editing `setup_db.py` and re-running it.

## Architecture

### Request Flow

```
User asks coach question
  → POST /api/chat  {message, board: [9-element list]}
  → analyze_board()     # rule-based: detects wins, blocks, fork traps
  → retrieve_from_pinecone()  # embeds query → top-2 Pinecone matches (score > 0.5)
  → build system prompt (role + board state + tactical analysis + RAG context)
  → Gemini generate_content()
  → return {"reply": "..."}
```

### Key Files

| File | Role |
|---|---|
| `main.py` | FastAPI app: `ChatRequest` model, `analyze_board()`, `retrieve_from_pinecone()`, `/api/chat` endpoint |
| `setup_db.py` | One-time script: embeds expert tips → upserts to Pinecone |
| `index.html` | All frontend: board UI, game logic, difficulty selector, Minimax (client-side JS), chat widget |
| `vercel.json` | Routing: `/api/*` → `main.py`, everything else → `index.html` |

### Board Representation

9-element list, index layout:
```
0 1 2
3 4 5
6 7 8
```
`"X"` = player, `"O"` = computer, `""` = empty.

### AI Difficulty

- **Easy/Medium:** Simple heuristic (JS, client-side)
- **Hard:** Full Minimax (JS, client-side) — runs in-browser, no backend involvement

## Deployment

Vercel deployment is the production target. The `vercel.json` routes `/api/*` to the Python backend and serves `index.html` as the static frontend. No build pipeline required.

## Tests

There are no automated tests in this project. Manual testing: start the server and exercise the game and chat UI in a browser.
