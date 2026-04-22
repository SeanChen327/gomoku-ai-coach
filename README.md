# Gomoku AI Coach (15x15)

A production-grade **15x15 Gomoku AI Coaching System** that combines a deterministic heuristic engine with Large Language Models (Gemini Pro) and RAG (Retrieval-Augmented Generation) to provide real-time tactical analysis.

This project is built with a "Quality First" mindset, featuring a robust CI/CD pipeline, AI output guardrails, and comprehensive automated testing.

---

## Tech Stack

- **Backend**: Python 3.10 (FastAPI), SQLAlchemy (PostgreSQL/SQLite)
- **AI Engine**: Custom Heuristic Search Algorithm + Gemini Pro (via LangChain)
- **Vector Database**: Pinecone (for RAG-based coaching knowledge)
- **Quality Assurance**: Pytest (Unit/Integration), Playwright (E2E), Locust (Load Testing)
- **CI/CD & DevOps**: GitHub Actions, Bandit (Security SAST), Render (Hosting)

---

## Key Features

- **Deterministic Heuristic Engine**: Precise pattern recognition for Gomoku shapes (Open-Four, Broken-Four, Open-Three, etc.).
- **AI Governance & Guardrails**: Integrated adversarial input detection and output validation to prevent "Coordinate Hallucination" (e.g., Z99).
- **Automated QA Pipeline**: Every PR triggers a full suite of integrity checks, including security scans and E2E browser simulations.
- **System Resilience**: Automated health checks and data retention cleanup via GitHub CRON triggers.

---

## Getting Started (for Developers)

### 1. Prerequisites

- **Python**: v3.10.14 (Strictly enforced for environment consistency).
- **Git**: For version control.

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/SeanChen327/gomoku-ai-coach.git](https://github.com/SeanChen327/gomoku-ai-coach.git)
cd gomoku-ai-coach

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install Playwright browsers (for E2E tests)
playwright install chromium --with-deps
```

### 3. Environment & Security Configuration

Copy the `.env.example` to `.env` and configure your keys. **All security keys are fully customizable** and must be updated for production environments.

| Variable           | Description     | Customization & Security                                                                                                |
| :----------------- | :-------------- | :---------------------------------------------------------------------------------------------------------------------- |
| `SECRET_KEY`       | JWT Signing Key | **Customizable.** Use a strong random string. Changing this will invalidate all active user sessions.                   |
| `CRON_SECRET`      | M2M Auth Token  | **Customizable.** This **MUST** match the `CRON_SECRET` stored in your GitHub Actions Secrets to allow automated tasks. |
| `GEMINI_API_KEY`   | Google AI API   | Obtain from Google AI Studio.                                                                                           |
| `PINECONE_API_KEY` | Vector DB API   | Obtain from Pinecone Dashboard.                                                                                         |

### 4. Running the Application

uvicorn main:app --reload

### 5. Testing Suite

We maintain a 95%+ logic coverage. Run tests using:

- **Core Logic**: pytest tests/test_ai_engine.py
- **API Integrity**: pytest tests/test_api_integration.py
- **End-to-End (E2E)**: pytest tests/test_e2e_frontend.py
- **AI Governance**: pytest tests/test_ai_governance.py
- **Load Testing**: locust -f tests/load_testing/locustfile.py

### 6. Governance & Contributions

To ensure system stability, we follow a strict Reviewer Protocol:

- **Branching**: All work must be done on feature/ branches. Direct commits to main are blocked.
- **Reviewers**: All Pull Requests (PRs) require approval from Ruby (@xxandy-what).
- **Consultancy**: For architectural or security changes, consult Sean (@SeanChen327).
- **CI Gate**: PRs can only be merged if the PR Integrity Check pipeline passes successfully.
