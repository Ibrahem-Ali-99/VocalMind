# 🧠 VocalMind

**VocalMind** is a modular AI ecosystem designed to bridge the gap between **voice research** and **production applications**. It integrates advanced speech processing (ASR, Diarization, Synthesis) with context-aware intelligence (RAG) to create next-generation conversational agents.

---

## Architecture

| Component    | Directory            | Tech Stack                  | Description                                                     |
| :----------- | :------------------- | :-------------------------- | :-------------------------------------------------------------- |
| **Backend**  | `backend/`           | FastAPI, SQLModel, Postgres | Central API gateway with auth, emotion, transcription endpoints |
| **Frontend** | `frontend/`          | Vite, React, Tailwind, MUI  | Manager and agent dashboards with analytics                     |
| **VAD**      | `services/vad/`      | Silero VAD, FastAPI         | Voice Activity Detection microservice (CPU-only)                |
| **WhisperX** | `services/whisperx/` | WhisperX, FastAPI           | Automatic Speech Recognition microservice                       |
| **Emotion**  | `services/emotion/`  | Transformers, FastAPI       | Speech emotion recognition microservice                         |
| **RAG**      | `services/rag/`      | LlamaIndex, Qdrant, Groq    | Retrieval-Augmented Generation for knowledge queries            |
| **Research** | `research/`          | Jupyter                     | Experiments for ASR, diarization, emotion, voice gen            |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with [uv](https://github.com/astral-sh/uv)
- **Node.js 20+**
- **Docker & Docker Compose**

### 1. Start All Services

```bash
make up          # Starts DB, Ollama, Qdrant, VAD, Emotion, WhisperX
```

### 2. Run Backend (Dev Mode)

```bash
make backend-install
make backend-dev     # → http://localhost:8000
```

### 3. Run Frontend (Dev Mode)

```bash
make frontend-install
make frontend-dev    # → http://localhost:3000
```

### 4. Run Everything with Docker

```bash
docker compose up --build
```

---

## 📁 Project Structure

```
VocalMind/
├── backend/          # FastAPI API gateway
├── frontend/         # Vite + React dashboard
├── services/         # Microservices (VAD, WhisperX, Emotion, RAG)
├── infra/            # Infrastructure (DB init, scripts)
├── research/         # Jupyter notebooks & experiments
├── docs/             # Documentation
├── .github/          # CI/CD workflows
├── docker-compose.yml
├── Makefile
└── CONTRIBUTING.md
```

---

## 🧪 Development

```bash
make backend-test    # Run backend tests
make backend-lint    # Lint with Ruff
make frontend-build  # Build frontend
make clean           # Remove caches
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

## License

[MIT](LICENSE)
