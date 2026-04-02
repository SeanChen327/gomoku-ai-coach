# Project Governance & Decision Log

## Feature: 4x4 Board Expansion

**Date:** 2026-04-02
**Branch:** `feature/board-4x4-expansion`
**Status:** Pending Peer Review

### 1. Technical Decisions

- **Algorithm Optimization**: Upgraded Minimax to include **Alpha-Beta Pruning** and a **Depth Limit (5)**. This was necessary to prevent browser hang-ups caused by the $4^4$ state space complexity of a 16-cell board.
- **Coordinate Mapping**: Shifted the win-condition logic from 8 patterns (3x3) to 10 patterns (4x4).
- **RAG Calibration**: Updated the Pinecone vector database. Deprecated 3x3 heuristics (e.g., center index 4) in favor of 4x4 strategic positioning (indices 5, 6, 10, 11).

### 2. Security & Quality Audit

- **Prompt Engineering**: Refined system prompts to strictly enforce Player X's perspective and prevent the AI from suggesting occupied or illegal moves.
- [cite_start]**Code Quality**: All Python functions now include Google-style Docstrings for better maintainability[cite: 1].
- [cite_start]**Environment**: API keys are securely managed via `.env` and are excluded from version control via `.gitignore`[cite: 1, 2].

### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (xxandy-what)
- **Technical Consultant**: Sean (hyu010)
