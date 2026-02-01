# 🧠 VocalMind

**VocalMind** is a modular AI ecosystem designed to bridge the gap between **voice research** and **production applications**. It integrates advanced speech processing (ASR, Diarization, Synthesis) with context-aware intelligence (RAG) to create next-generation conversational agents.

---

## Component 

This repository is a monorepo containing both production-ready services and research experiments.
| Module | Type | Tech Stack | Description |
| :-------------- | :---------- | :------------------------- | :------------------------------------------------------------------------- |
| **RAG System** | Application | LlamaIndex, Groq, Pinecone | Fully functional RAG app with local embeddings (Ollama) and vector search. |
| **Backend** | Service | FastAPI, Uvicorn, UV | Project shell initialized; scaffolding ready for API implementation. |
| **ASR** | Research | Jupyter | Notebooks for Automatic Speech Recognition experiments. |
| **Diarization** | Research | NVIDIA NeMo, Jupyter | Speaker diarization and clustering experiments. |
| **Voice Gen** | Script | Python | Experimental scripts for TTS and voice synthesis. |
| **Frontend** | Client | React, Node.js | Initial configuration present. |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** (Required for dependency management)
- **Docker** (For local RAG embeddings)

### 1. 📚 Running the RAG System 

The RAG module is currently the most active component.

1. **Navigate to the module:**
   ```bash
   cd Experiments/Rag
   ```
2. **Setup Environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your GROQ_API_KEY and PINECONE_API_KEY
   ```
3. **Install & Run Services:**
   ```bash
   uv sync
   docker compose up -d  # Starts local Ollama instance
   ```
4. **Interact:**
   ```bash
   uv run python -m rag_app.main
   ```

### 2. 🧪 Experiments Modules

To explore the notebooks for ASR or Diarization:

1. **Install root dependencies:**
   ```bash
   uv sync
   ```
2. **Launch Jupyter:**
   ```bash
   uv run jupyter lab
   ```
   _Open the notebooks in `Experiments/Automatic-Speech-Recognition` or `Experiments/Speaker-Diarization`._

### 3. 🔌 Backend Development

The backend is initialized for future API development.

```bash
cd backend
uv sync
uv run uvicorn main:app --reload
```

---
