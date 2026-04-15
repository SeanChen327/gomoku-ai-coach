# Project Governance & Decision Log

## 📌 Project Workflows & Review Standards

To ensure production-grade quality and security, all contributors must strictly adhere to the following protocol:

- **Branching Strategy**: All features must be developed on isolated branches (`feature/<feature-name>`). Direct commits to `main` are strictly prohibited.
- **Pull Request (PR) Protocol**: Every PR must include a clear description of the technical changes and any mock data/tests used.
- **Mandatory Reviewers**:
  - **Ruby (@xxandy-what)** MUST act as the Primary Peer Reviewer. No code is merged without her explicit approval.
  - **Sean (@SeanChen327)** MUST be tagged as the Technical Consultant for architectural and security decisions.

---

## 📖 Decision Log

### Feature: Rich CSV Reporting & Heuristic Enhancement

**Date:** 2026-04-14
**Branch:** `feature/rich-csv-reporting`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Heuristic Comparison Logic**: implemented `evaluatePlayerMove` in `index.html` to compare the player's move score against the optimal move score calculated by the frontend engine.
- **Dynamic Metadata Generation**: Introduced qualitative labels (`Blunder`, `Critical`, `Strong Move`, etc.) and contextual comments to replace static coordinate strings in the CSV export.
- **Missed Opportunity Tracking**: Added logic to capture and log the `Missed_Best_Move` coordinate when a player ignores a critical tactical threat (score >= 10000).

#### 2. Security & Quality Audit

- **Zero API Modification**: Mathematically guaranteed no changes to `main.py` or existing REST API contracts.
- **State Integrity**: Used temporary board cloning during heuristic evaluation to prevent state pollution in the active game.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: Hybrid AI Algorithm for Easy Mode

**Date:** 2026-04-14
**Branch:** `feature/easy-mode-ai`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Epsilon-Greedy Logic Implementation:** Modified the "Easy" difficulty algorithm in the frontend engine. Introduced `getEasyMove()`, replacing pure randomness (`getRandomMove()`).
- **Threat-Blocking Enforcement:** Ensured that if the evaluation score exceeds a critical threshold (10000, indicative of immediate loss or win scenarios), the AI will always execute the optimal defensive/offensive move.
- **Sub-Optimal Blending:** During non-critical board states, forced a 60/40 probability split between random moves and optimal heuristic moves to simulate an entry-level player strategy.

#### 2. Security & Quality Audit

- **Zero API Modification:** Guaranteed absolute isolation from the backend. The core `main.py` files and REST APIs remain untouched.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: Heuristic Payload Injection & Frontend Computation Offloading

**Date:** 2026-04-13
**Branch:** `feature/ai-chat-heuristics`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Computation Offloading**: Relocated Gomoku spatial calculations (Win Rate estimation and Best Next Move prediction) entirely to the frontend (`index.html`) using a lightweight JavaScript heuristic engine[cite: 1].
- **Dynamic Payload Injection**: Bypassed the need to alter backend API contracts by packaging frontend calculations into the existing `last_evaluation` JSON dictionary. This payload is passed seamlessly to the backend during the `fetch` call.
- **Role Reassignment for LLM**: Shifted the Gemini model's responsibility from _calculating_ coordinates to _explaining_ them. The prompt now strictly enforces reliance on the injected frontend payload, drastically improving response accuracy and tactical value.

#### 2. Security & Quality Audit

- **Zero API Intrusion**: Mathematically guaranteed that no core application endpoints, database models, or API deployment configurations in `main.py` were broken or modified.
- **Cost & Latency Reduction**: Eliminates the need for backend Python to process a 225-cell array for every chat query. By handling math in the browser, we save compute cycles and reduce LLM token generation (enforced `<80 words` limit).
- **Hallucination Mitigation**: Prevented the AI from blindly guessing spatial coordinates, anchoring its advice entirely in deterministic algorithmic math.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: RenjuNet RAG Knowledge Base Integration

**Date:** 2026-04-13
**Branch:** `feature/rag-renjunet-upgrade`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Data Sourcing**: Migrated the RAG knowledge base to utilize professional Gomoku/Renju strategies from The International Renju Federation (RenjuNet).
- **Bypass Data Pipeline**: Engineered isolated `scrape_renjunet.py` and `ingest_renjunet.py` scripts to collect, clean (JSON), and embed data into Pinecone.
- **Zero API Intrusion**: Strictly maintained existing API contracts in `main.py`; no core application endpoints or prompt injection code were modified.

#### 2. Security & Quality Audit

- **Data Quality**: Utilized a JSON intermediary layer to allow manual inspection and filtering of scraped web data before vectorization, drastically reducing the risk of LLM hallucinations.
- **Environment Isolation**: Operational scripts securely load `GEMINI_API_KEY` and `PINECONE_API_KEY` directly from the local environment, avoiding credential leakage.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: CSV Standardization & UX Flow Optimization

**Date:** 2026-04-13
**Branch:** `feature/csv-data-and-ux-fix`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Data Mapping Upgrade**: Standardized CSV output by mapping internal symbols (X/O) to professional Go/Gomoku terminology (Black/White).
- **Coordinate System Transformation**: Implemented a conversion formula in `downloadCSV()` to translate 1D array indices (0-224) into 2D algebraic coordinates (A-O, 1-15).
- **UX Interruption Fix**: Decoupled the AI Report generation from the game-end event. Replaced the auto-modal with a "View AI Report" button in the status bar to allow users to inspect the board before navigating away.

#### 2. Security & Quality Audit

- **API Integrity**: Verified that the frontend-to-backend data contract (`history` object) remains unchanged to avoid breaking existing FastAPI endpoints.
- **State Consistency**: Ensured the game-end result string is passed correctly to the manual trigger button.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: Gomoku 15x15 UI/UX Optimization (Coordinates & Grid Fix)

**Date:** 2026-04-13
**Branch:** `feature/board-coordinates-ui`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Coordinate System Implementation**: Added a CSS Grid-based coordinate wrapper (A-O, 1-15) to the game board. This ensures users can accurately communicate move locations to the AI Coach.
- **Visual Logic Refactoring (Intersections)**: Corrected the stone placement logic. Stones are now visually rendered at the **intersections** of grid lines rather than inside cells, adhering to standard Gomoku/Go rules.
- **Grid Bleeding Fix**: Utilized CSS `::before` pseudo-elements to strictly constrain grid lines within the 14x14 interior play area, eliminating line overflow at the board edges.
- **Zero-Logic Intrusion**: Verified that all UI changes are localized to CSS/HTML and do not impact the existing FastAPI backend or Gemini/Pinecone integration contracts.

#### 2. Security & Quality Audit

- **Responsiveness**: Maintained fixed dimensions (525px) for the board to ensure consistent coordinate alignment across different browser viewports.
- **Event Delegation**: Maintained the 35px click-box size for `.cell` elements to ensure high touch/click tolerance while visually snapping stones to intersections.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: Gomoku 15x15 Migration & Heuristic AI

**Date:** 2026-04-13
**Branch:** `feature/gomoku-core-logic`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Game Engine Overhaul**: Transitioned from a 4x4 Tic-Tac-Toe grid to a 15x15 Gomoku (Five-in-a-Row) board.
- **AI Algorithm Optimization**: Replaced the Minimax algorithm with a **Heuristic Evaluation Engine**. This prevents the exponential state-space explosion ($225!$) that would occur on a 15x15 grid, ensuring sub-second response times in the browser.
- **UI/UX Enhancement**: Implemented a wood-textured board with realistic black and white stone rendering for a more authentic Gomoku experience.
- **Tactical Analysis Engine**: Upgraded the backend `analyze_board` function to recognize Gomoku-specific patterns like "Open Threes" and "Critical Fours" to provide accurate coaching feedback via Gemini.

#### 2. Security & Quality Audit

- **State Integrity**: Maintained consistent API contracts to ensure that existing Gemini and Pinecone integrations function without modification.
- **Audit Logging**: Maintained strict [SECURITY EVENT] logging in the backend for all game-related actions.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

### Feature: Automation Keep-Alive Mechanism (Strategy B)

**Date:** 2026-04-05
**Branch:** `feature/keep-alive-strategy-b`
**Status:** Pending Peer Review

#### 1. Technical Decisions

- **Infrastructure Optimization**: Implemented a "Keep-Alive" strategy to mitigate Render's free-tier hibernation policy, which typically causes a 50-60 second cold start delay after 15 minutes of inactivity.
- **Health Check Endpoint**: Developed a lightweight `/api/health` endpoint in `main.py` that returns minimal JSON (status and timestamp). This design ensures that periodic pings do not trigger heavy database or AI logic, maintaining near-zero resource consumption.
- **External Orchestration**: Recommended the use of an external uptime monitoring service (e.g., UptimeRobot) to trigger the health check endpoint every 10-15 minutes, ensuring the instance remains in an "Active" state.

#### 2. Security & Quality Audit

- **Resource Efficiency**: Verified that the health check route avoids SQLAlchemy sessions and Gemini API calls, preventing unnecessary infrastructure load.
- **Security Perimeter**: The `/api/health` route is public and read-only. It does not expose environment variables, user data, or sensitive system internals.

#### 3. Review Protocol

- **Primary Peer Reviewer**: Ruby (@xxandy-what)
- **Technical Consultant**: Sean (@SeanChen327)

---

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
