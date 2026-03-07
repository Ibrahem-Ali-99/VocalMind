# VocalMind

VocalMind is a modular AI ecosystem integrating speech processing (ASR, Diarization, Synthesis) with retrieval-augmented generation (RAG) to create context-aware conversational agents, designed for call center and telecom use cases.

---

## Architecture

| Component    | Tech Stack | Description |
| :----------- | :--------- | :---------- |
| **Backend**  | FastAPI, SQLModel, asyncpg | Central API gateway with auth (JWT/Google OAuth), Supabase integration, and dispute handling. |
| **Frontend** | React 18, Vite, Tailwind v4, MUI, Radix UI | Manager and agent dashboards with session analysis. Tested with Cypress E2E and Vitest. |
| **VAD**      | Silero VAD, FastAPI | Voice Activity Detection microservice. |
| **WhisperX** | WhisperX, pyannote, FastAPI | Automatic Speech Recognition and Diarization microservice. |
| **Emotion**  | Transformers, FastAPI | Speech emotion recognition microservice. |
| **RAG**      | LlamaIndex, Qdrant, Groq, Ollama | Retrieval-Augmented Generation for knowledge queries. |
| **Ingestion**| LlamaIndex | Automated pipeline for RAG document ingestion. |
| **Research** | Jupyter | Reference experiments for speech pipelines and voice generation. |

---

## Quick Start

### Prerequisites

- **Python 3.12+** (via [uv](https://github.com/astral-sh/uv))
- **Node.js 20+**
- **Docker & Docker Compose**

### Configuration
Copy `.env.example` to `.env` and fill in the required api keys:
```bash
cp .env.example .env
```
Key requirements: Groq, Google, Pinecone, ElevenLabs, HuggingFace.

### 1. Start Services (Docker)

Start all services (Database, Backend, Frontend, Ollama, Qdrant, Ingestion, VAD, Emotion, WhisperX):

```bash
make up
```

### 2. Run Local Development

**Backend:**
```bash
make backend-install
make backend-dev     # → http://localhost:8000
```

**Frontend:**
```bash
make frontend-install
make frontend-dev    # → http://localhost:3000
```

---

## Project Structure

```text
VocalMind/
├── backend/          # FastAPI API gateway
├── frontend/         # React dashboard (Manager & Agent routes)
├── services/         # Microservices (VAD, WhisperX, Emotion, RAG)
├── infra/            # DB initialization and python seed scripts
├── research/         # Jupyter notebooks & prototype scripts
├── data/             # General data and reference materials
├── docker/           # Shared Docker configuration artifacts
├── docs/             # Documentation assets
├── AudioData/        # Sample audio inputs for testing
├── .github/          # CI workflows (ci.yml, backend.yml, frontend.yml, rag_ci.yml)
├── docker-compose.yml# Multi-container service definitions
├── Makefile          # Unified development commands
└── CONTRIBUTING.md   # Guidelines
```

---

## Useful Commands

### Backend
```bash
make backend-install  # Install dependencies
make backend-dev      # Run api gateway
make backend-test     # Run pytest suite
make backend-lint     # Run Ruff linter
```

### Frontend
```bash
make frontend-install # Install dependencies
make frontend-test    # Run Cypress E2E tests
make frontend-lint    # Run ESLint validation
make frontend-build   # Build production bundle
```

### Docker
```bash
make up               # Start all services
make logs             # Tail container logs
make build            # Rebuild images
make down             # Stop all services
```

### General
```bash
make clean            # Remove caches and build artifacts
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## License

[MIT](LICENSE)
