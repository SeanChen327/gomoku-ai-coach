# Project Governance & Decision Log

## 📌 Project Workflows & Review Standards

To ensure production-grade quality and security, all contributors must strictly adhere to the following protocol:

- **Branching Strategy**: All features must be developed on isolated branches (`feature/<feature-name>`). Direct commits to `main` are strictly prohibited.
- **Pull Request (PR) Protocol**: Every PR must include a clear description of the technical changes and any mock data/tests used.
- **Mandatory Reviewers**:
  - **Ruby (@xxandy-what)** MUST act as the Primary Peer Reviewer. No code is merged without her explicit approval.
  - **Sean (@hyu010)** MUST be tagged as the Technical Consultant for architectural and security decisions.

---

## 📖 Decision Log

### Feature: Cloud Native Deployment Architecture (Render + PostgreSQL)

**Date:** 2026-04-04
**Branch:** `feature/render-deployment`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Infrastructure as Code (IaC)**: Introduced `render.yaml` to orchestrate cloud resources autonomously, including a managed PostgreSQL instance and a Python Web Service container.
- **Database Engine Migration**: Upgraded SQLAlchemy configuration to dynamically detect and switch from local `sqlite://` to Render's `postgresql://`, ensuring data persistence across ephemeral container deployments. Added `psycopg2-binary` driver.
- **Unified Full-Stack Routing**: Deprecated Vercel's static routing model. Bound FastAPI's root endpoint (`/`) to explicitly serve `index.html` via `FileResponse`, allowing Render to act as a unified full-stack host.

#### 2. Security & Quality Audit

- **CORS Strategy**: Maintained wildcard `["*"]` for initial deployment flexibility, flagged for restriction to the strict Render app URL in the next production hotfix.
- **Environment Isolation**: Configured Render environment variables to strictly reject syncing of `GEMINI_API_KEY` and `PINECONE_API_KEY` into public blueprints, forcing secure Dashboard injection.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: Database Persistence & User Registration System

**Date:** 2026-04-04
**Branch:** `feature/database-integration`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **ORM & Database Engine**: Transitioned from a hardcoded `MOCK_USERS_DB` to a robust relational database model using **SQLAlchemy** and **SQLite** (`tictactoe.db`). This serves as a highly portable local development setup, ensuring a seamless future migration path to PostgreSQL in production.
- **Registration Endpoint**: Engineered a new `/api/register` endpoint enforcing strict validation (Pydantic `EmailStr`) to prevent duplicate usernames and emails.
- **Frontend Auth UI Evolution**: Upgraded the authentication state machine in `index.html` to support a unified Auth Modal with smooth toggling between "Login" and "Sign Up" views. Successfully decoupled the UI from the mock data limitation.

#### 2. Security & Quality Audit

- **Password Cryptography**: Maintained strict enforcement of `bcrypt` via `passlib` for all new user registrations. Raw passwords are mathematically guaranteed to never touch the database or logs.
- **Audit Logging**: Enhanced server-side logging (`logger.info`) to explicitly track the exact identity and lifecycle events (`[SECURITY EVENT] New user registered successfully`).

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: User Authentication & Zero-Trust API Lockdown

**Date:** 2026-04-03
**Branch:** `feature/user-authentication`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Authentication Standard**: Implemented OAuth2 password flow with JWT (JSON Web Tokens) to secure backend resources.
- **Zero-Trust UI**: Engineered a strict UI state machine. The game board and AI interfaces are completely hidden (`display: none`) until a valid JWT is detected in `localStorage`.
- **API Lockdown**: Secured FastAPI endpoints (`/api/chat`, `/api/generate-report`) using `Depends(get_current_active_user)`, mathematically guaranteeing zero unauthorized API quota consumption.
- **Vercel Deployment Hotfix**: Explicitly pinned `bcrypt<4.0.0` in `requirements.txt`. This resolved a critical serverless initialization crash (`ValueError: password cannot be longer than 72 bytes` and `AttributeError`) caused by `passlib` failing its internal wrap bug detection against `bcrypt 4.x`.

#### 2. Security & Quality Audit

- **Audit Logging**: Injected explicit server-side logging (`logger.info`) to track the exact identity of users triggering expensive LLM calls.
- **Environment Resiliency**: Upgraded `API_BASE_URL` resolution in `index.html` to natively support offline `file:///` protocol testing without triggering CORS/Network errors.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: AI Game Report System (Hybrid Intelligence Architecture)

**Date:** 2026-04-02
**Branch:** `feature/ai-game-report`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Hybrid Intelligence Architecture**: Migrated from pure LLM analysis to a "Frontend Heuristic Tagging + Backend LLM Summarization" model. The frontend Minimax algorithm now silently tags every move in real-time (`evaluation_label`, `missed_best_move`), drastically reducing LLM hallucination and latency.
- **Dual-Format Data Export**: Established JSON as the application's internal source of truth, with a seamless frontend engine to compile this JSON into a downloadable CSV report for users instantly at the end of the match.
- **Environment Auto-Detection**: Implemented dynamic configuration for `API_BASE_URL` in `index.html`. It automatically switches between `http://127.0.0.1:8000` (Local) and relative paths `""` (Vercel Production) based on `window.location.hostname`.

#### 2. Security & Quality Audit

- **State Race Condition Fix**: Resolved a critical bug where the AI Coach was analyzing the computer's ('O') immediate response instead of the user's ('X') move. Replaced basic array popping with a reverse human-player search (`reverse().find(m => m.player === "X")`).
- **Data Contract Upgrade**: Safely upgraded the `Move` and `ChatRequest` Pydantic models in `main.py` using `Optional[Dict]` to accept the new heuristic tags without breaking backward compatibility or triggering `422 Unprocessable Entity` errors.
- **Rate Limit Mitigation**: Switched default LLM calls from `gemini-2.5-flash` to `gemini-2.0-flash` (or `1.5-flash`) for local development to bypass the strict 20 RPM Free Tier limits.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: 4x4 Board Expansion

**Date:** 2026-04-02
**Branch:** `feature/board-4x4-expansion`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Algorithm Optimization**: Upgraded Minimax to include **Alpha-Beta Pruning** and a **Depth Limit (5)**. This was necessary to prevent browser hang-ups caused by the $4^4$ state space complexity of a 16-cell board.
- **Coordinate Mapping**: Shifted the win-condition logic from 8 patterns (3x3) to 10 patterns (4x4).
- **RAG Calibration**: Updated the Pinecone vector database. Deprecated 3x3 heuristics (e.g., center index 4) in favor of 4x4 strategic positioning (indices 5, 6, 10, 11).

#### 2. Security & Quality Audit

- **Prompt Engineering**: Refined system prompts to strictly enforce Player X's perspective and prevent the AI from suggesting occupied or illegal moves.
- **Code Quality**: All Python functions now include Google-style Docstrings for better maintainability.
- **Environment**: API keys are securely managed via `.env` and are excluded from version control via `.gitignore`.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)
