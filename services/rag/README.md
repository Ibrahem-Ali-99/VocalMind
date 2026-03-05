# Generic RAG System

A flexible Retrieval-Augmented Generation (RAG) system built with [LlamaIndex](https://www.llamaindex.ai/), [Groq](https://groq.com), and [Pinecone](https://www.pinecone.io). Designed to ingest documents from various sources, index them into a vector store, and provide accurately sourced answers using LLMs.

## Features

- **High-Performance LLM**: Uses Groq for ultra-fast inference.
- **Vector Search**: Pinecone for scalable similarity search.
- **Local Embeddings**: Uses Ollama for generating embeddings locally.
- **Pluggable Data Loaders**: Strategy pattern supports multiple data sources:
  - `standard` – Basic file ingestion (PDF, TXT, MD, etc.)
  - `directory` – Custom directory loader with sidecar metadata (`.meta.json`)
  - `herb` – HERB enterprise dataset (Slack, PRs, Meetings, Documents)
  - `ragbench` – HuggingFace RagBench datasets
- **Heuristic Metadata Extraction**: Fast, loader-based title/summary generation (no LLM calls during ingestion).
- **Detailed Logging**: Logs every query, including retrieval timing, chunks used, and the final response in JSON format.
- **Evaluation Framework**: Built-in tools to evaluate RAG accuracy using RAGAS and LLM-as-a-Judge.

## Setup

1. **Install Dependencies**
   Ensure you have Python 3.11+ installed.

   ```bash
   uv sync
   ```

   (Or use `pip install -r requirements.txt` if you export dependencies).

2. **Environment Configuration**
   Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   - **GROQ_API_KEY**: Your Groq API key.
   - **PINECONE_API_KEY**: Your Pinecone API key.
   - **PINECONE_INDEX_NAME**: (Optional) Name of your index.

3. **Start Local Embeddings (Ollama)**
   This project uses Ollama for local embeddings. You can start it easily with Docker:

   ```bash
   # Start in background
   docker compose up -d
   ```

   (Alternatively, install Ollama from [ollama.com](https://ollama.com) and run it manually).

## Usage

Run the module from the project root.

### 1. Ingest Documents

Process and index documents. Data loaders are configured in `rag_app/config.py`.

```bash
uv run python -m rag_app.main --ingest
```

Use `--force` to force a complete re-index:

```bash
uv run python -m rag_app.main --ingest --force
```

### 2. Interactive Search (Default)

Start an interactive chat session to query your documents.

```bash
uv run python -m rag_app.main
```

### 3. Single Query

Execute a quick single query and exit.

```bash
uv run python -m rag_app.main -q "What is the main topic?"
```

### 4. Evaluation

Run the evaluation suite to measure accuracy and performance. This uses the HERB dataset (or your custom data) to benchmark the RAG system.

```bash
# Run full evaluation (default limit: 10 samples)
uv run python -m rag_app.main --evaluate

# Run with custom limit
uv run python -m rag_app.main --evaluate --limit 20

# Filter by product/category
uv run python -m rag_app.main --evaluate --filter "ActionGenie"
```

Reports are generated in the `reports/` directory as JSON files, containing:

- Accuracy scores (LLM-as-a-Judge)
- Latency metrics (retrieval vs synthesis)
- Detailed Q&A logs with retrieved contexts

## Logs

Query logs are automatically saved to the `logs/` directory in JSON format. Each log includes:

- Timestamp & Model info
- Retrieval & Synthesis timing
- Retrieved text chunks with metadata (and scores)
- Final response
