# VocalMind RAG

**Dual-granularity Policy Compliance & Answer Scoring with Docling, Qdrant, and Groq**

A robust Retrieval-Augmented Generation (RAG) system for policy document analysis, featuring:

- Dual-granularity chunking (full sections & precision snippets)
- Policy compliance and answer correctness evaluation
- Fast, local-first architecture (Docling, Ollama, Qdrant, Groq)
- Per-organization document support

---

## Features

- **Dual Collection RAG**: Ingests policy PDFs into two Qdrant collections:
	- **Parents**: Full policy sections (for compliance checks)
	- **Children**: Fine-grained snippets (for answer fact-checking)
- **AI-powered PDF Parsing**: Uses Docling for accurate PDF-to-Markdown conversion
- **Local Embeddings**: Embeddings generated via Ollama (snowflake-arctic-embed2)
- **Fast LLM Synthesis**: Uses Groq for rapid, high-quality LLM responses
- **Per-Organization Filtering**: Each org’s docs are indexed with metadata for targeted queries
- **Policy Compliance & Answer Scoring**: Built-in evaluators for both transcript compliance and answer correctness
- **Detailed Logging**: All queries and evaluations are logged as JSON in `logs/`
- **Evaluation Suite**: Automated scoring and reporting for compliance and answer accuracy
- **Explainability Hooks**: Retrieval provenance can be surfaced to manager UI and API consumers

## Documentation

Team-facing RAG documentation is maintained in:

1. `docs/rag/RAG_OVERVIEW.md`
2. `docs/rag/INGESTION_PIPELINE.md`
3. `docs/explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md`

---

## Setup

1. **Install Dependencies**
	 - Requires Python 3.11+
	 - Install with:
		 ```bash
		 pip install -r requirements.txt
		 ```
		 (Or use `uv sync` if using uv)

2. **Environment Configuration**
	 - Copy `.env.example` to `.env` and fill in:
		 - `GROQ_API_KEY` (Groq LLM)
		 - `QDRANT_URL` (Qdrant vector DB, default: `http://localhost:6333`)
		 - `OLLAMA_BASE_URL` (Ollama embeddings, default: `http://localhost:11434`)

3. **Prepare Policy/SOP Documents**
	 - Place PDFs under `sop-standards/`:
		 ```
		 sop-standards/
		 └── org1/
			 ├── policy-docs/
			 │   ├── policy_a.pdf
			 │   └── policy_b.pdf
			 └── sop-procedures/
				 └── sop_a.pdf
		 ```
	 - Parsed markdown outputs are written to `sop-standards/{org}/parsed-docs/`.

4. **Start Services**
	 - Ensure Qdrant and Ollama are running (see their docs for Docker or local start).

---

## Usage

Run from the `final-rag` directory:

### 1. Ingest Documents

```bash
python main.py --ingest
```
Use `--force` to wipe and re-index:
```bash
python main.py --ingest --force
```

### 2. Querying

- **Single Query:**
	```bash
	python main.py -q "What is the refund policy?" --org org1
	```
- **Interactive Mode:**
	```bash
	python main.py
	```

### 3. Policy Compliance Check

Evaluate if a transcript complies with policy:
```bash
python main.py --compliance "The agent promised a full refund with no questions asked." --org org1
```

### 4. Answer Correctness Check

Check if an agent’s answer is factually correct:
```bash
python main.py --check-answer --question "Refund window?" --answer "30 days" --org org1
```

---

## Evaluation & Logs

- All queries and evaluations are logged in the `logs/` directory as JSON.
- Each log includes:
	- Timestamp, model info, org, and query type
	- Retrieved chunks and metadata
	- LLM response and evaluation scores

---

## Notes

- The ingestion pipeline discovers PDFs in `sop-standards/{org}/policy-docs` and `sop-standards/{org}/sop-procedures`.
- Per-organization filtering is enabled via the `--org` flag.
- For compliance and answer checks, replace the quoted text with your own queries or transcripts.

---

## Testing

Unit tests are provided for all core RAG modules (ingestion, query engine, evaluator, config). Tests are located in the `tests/` directory and cover:
- Ingestion pipeline logic (cleaning, chunking, metadata)
- Query engine node conversion and log writing
- Evaluator result models and JSON parsing
- Config and settings validation

### Running Tests

You can run all tests with:

```bash
uv run pytest tests/ -v --tb=short
```
Or, if not using `uv`:
```bash
pytest tests/ -v --tb=short
```

All tests are unit tests and do not require external services or network access.
