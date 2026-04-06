# RAG Overview

## Purpose

The RAG service provides policy-grounded retrieval and answer synthesis for compliance and QA workflows.

It combines:
1. Docling PDF parsing
2. Local embeddings (Ollama)
3. Qdrant vector search (dual collections)
4. Groq LLM synthesis

It also now feeds the Evidence-Anchored Explainability Layer used in manager call review.

## Key Concepts

1. Dual-granularity indexing
- Parents collection: full policy sections for compliance context
- Children collection: compact snippets for answer fact-checking

2. Per-organization isolation
- Documents are discovered by org folder under `sop-standards/{org}`
- Retrieval can be filtered by org metadata

3. Query modes
- Compliance mode queries parent chunks
- Answer mode queries child chunks

4. Retrieval provenance
- Retrieved chunks can be surfaced with similarity, reference path, and lightweight verdict metadata.
- Interaction detail pages expose this through `claimProvenance`.
- The standalone RAG route exposes `retrieval_provenance`.

## Main Runtime Components

1. `services/rag/query_engine.py`
- Embeds query text via Ollama
- Queries Qdrant with optional org filter
- Converts results to LlamaIndex nodes
- Synthesizes final response using Groq
- Logs query traces in `services/rag/logs`

2. `services/rag/evaluator.py`
- Structured scoring helpers for compliance and answer quality

3. `services/rag/config.py`
- Central settings for model, endpoints, collections, and paths

4. `backend/app/api/routes/rag.py`
- Wraps RAG results for the frontend API.
- Returns `retrieval_provenance` on `/api/v1/rag/query`.

## Collections

1. Parents
- Name: `vocalmind_parents` (default)
- Content: larger policy sections
- Usage: policy-level interpretation and consistency checks

2. Children
- Name: `vocalmind_children` (default)
- Content: shorter precision snippets
- Usage: pinpoint factual grounding for answer checks

## Directory Layout (Current)

1. Source docs
- `sop-standards/{org}/policy-docs/*.pdf`
- `sop-standards/{org}/sop-procedures/*.pdf`

2. Parsed markdown outputs
- `sop-standards/{org}/parsed-docs/*.md`

3. Pipeline report
- `sop-standards/_pipeline_report.json`

## Config Summary

1. `DOCS_DIR`
- Base source root (default: `sop-standards`)

2. `PARSED_DIR`
- Base output root for parsed markdown + pipeline report (default: `sop-standards`)

3. `QDRANT_URL`
- Vector DB endpoint

4. `OLLAMA_BASE_URL` / `EMBEDDING_MODEL`
- Embedding provider settings

5. `GROQ_API_KEY`
- LLM synthesis provider

## Typical Flow

1. Ingestion parses PDFs and indexes vectors
2. Query engine retrieves relevant chunks
3. Groq synthesizes response from retrieved context
4. Retrieval provenance is attached for explainability and auditability
5. Logs and timing are stored for auditability

## Testing

Primary tests:
1. `services/rag/tests/test_ingest.py`
2. `services/rag/tests/test_query_engine.py`
3. `services/rag/tests/test_evaluator.py`
4. `services/rag/tests/test_config.py`
5. `backend/tests/test_interactions_llm_triggers.py`
6. `backend/tests/test_sop_retrieval.py`

Run:

```bash
cd services/rag
uv run pytest tests/ -v
```
