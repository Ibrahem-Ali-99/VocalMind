# LLM Trigger Feature Guide

## Purpose

The LLM Trigger feature evaluates customer-agent interactions and returns coaching/compliance signals in three dimensions:

1. Emotion Shift: text vs acoustic mismatch and dissonance analysis
2. Process Adherence: SOP step coverage and resolution quality
3. NLI Policy: contradiction/entailment check against policy context

This guide documents architecture, data flow, folder structure, runtime behavior, and testing so the team can maintain and extend the feature safely.

## High-Level Architecture

1. Backend interaction detail endpoint can request LLM trigger evaluation.
2. LLM trigger service loads transcript + context and runs analysis chains.
3. SOP context is resolved with priority order:
   - Manual SOP standards (organization-specific)
   - Qdrant retrieval fallback
4. Results are mapped into a frontend-friendly payload.
5. Manager and Agent views render diagnostics and coaching insights.

## Core Backend Files

1. `backend/app/api/routes/interactions.py`
- Exposes interaction detail endpoint with optional LLM trigger fields.
- Maps internal report to API payload with `_map_llm_trigger_report`.
- Resolves organization slug via `_resolve_llm_org_filter`.

2. `backend/app/llm_trigger/service.py`
- Main orchestration and heuristics.
- Runs chains for emotion/process/NLI.
- Contains rolling window helpers, evidence construction, deterministic guards.

3. `backend/app/llm_trigger/retrieval.py`
- SOP retrieval resolver.
- Manual SOP path takes precedence.
- Falls back to Qdrant retrieval.

4. `backend/app/llm_trigger/chains.py`
- LangChain chain builders for the three analysis tasks.

5. `backend/app/llm_trigger/prompts.py`
- Prompt templates and output constraints.

6. `backend/app/llm_trigger/schemas.py`
- Typed response contracts for internal LLM trigger objects.

## Frontend Files

1. `frontend/src/app/services/api.ts`
- Type definitions for `llmTriggers` payload.
- Interaction detail request supports LLM options.

2. `frontend/src/app/components/manager/SessionDetail.tsx`
- Renders manager-focused trigger diagnostics.
- Supports LLM refresh action.

3. `frontend/src/app/components/agent/AgentCallDetail.tsx`
- Renders coaching-focused trigger insights.
- Supports LLM refresh action.

## SOP and Policy Folder Structure

Canonical root:

`sop-standards/`

Per organization:

1. `sop-standards/{org}/policy-docs/*.pdf`
- Source policy PDFs.
- Consumed by RAG ingestion (Docling converts PDF -> markdown).

2. `sop-standards/{org}/sop-procedures/*.pdf`
- Source SOP PDFs from organizations.
- Also ingested and converted by Docling.

## Parsed Markdown and Runtime Consumption

1. RAG ingestion writes converted markdown into:
- `sop-standards/{org}/parsed-docs/*.md`

2. Backend SOP retrieval reads SOP context from parsed markdown:
- For each SOP PDF in `sop-procedures`, backend looks up matching stem in `parsed_docs`.
- Example:
  - source: `sop-standards/nexalink/sop-procedures/01-refund.pdf`
  - parsed: `sop-standards/nexalink/parsed-docs/01-refund.md`

3. If no parsed markdown exists, retrieval has backward compatibility fallback for direct text files in `sop-procedures` (`.md` / `.txt`).

## Config Keys

### Backend (`backend/app/core/config.py`)

1. `SOP_DOCS_ROOT`
- Default: `sop-standards`

2. `SOP_PARSED_DOCS_ROOT`
- Default: `sop-standards` (resolved as `sop-standards/{org}/parsed-docs`)

3. `SOP_RETRIEVAL_TOP_K`
- Used for Qdrant fallback retrieval.

4. `QDRANT_COLLECTION_SOP_PARENTS` and `QDRANT_COLLECTION_SOP_CHILDREN`
- Stores specialized SOP vectors separately from Policies to eliminate RAG cross-pollution.

### RAG (`services/rag/config.py`, `.env`)

1. `DOCS_DIR`
- Expected to point to `sop-standards`

2. `PARSED_DIR`
- Base output root for parsed markdown and pipeline report.

## Endpoint Behavior

Interaction detail supports these query options:

1. `include_llm_triggers=true`
- Request LLM trigger payload.

2. `llm_org_filter=<org>`
- Optional explicit org override.

3. `llm_force_rerun=true`
- Recompute (used by refresh controls).

If `llm_org_filter` is omitted, backend resolves org slug from interaction -> organization relation.

## Retrieval Priority

When process adherence analysis needs SOP context:

1. If supplied `retrieved_sop_from_pinecone` is non-empty, use it.
2. Else attempt manual SOP context from `sop-standards/{org}/sop-procedures` via parsed markdown.
3. Else query dedicated SOP Qdrant fallback (`SOPRetriever` pointing to `vocalmind_sop_parents`).

## Ingestion Behavior for PDF Discovery

RAG ingestion now scans per org for both folders:

1. `policy-docs`
2. `sop-procedures`

Legacy fallback remains for org root PDFs.

## Full Validation Command

Use Make target from repository root:

`make llm-trigger-test`

This runs:

1. Backend LLM-trigger tests:
- `tests/test_llm_trigger_service.py`
- `tests/test_interactions_llm_triggers.py`
- `tests/test_sop_retrieval.py`

2. RAG ingestion tests:
- `services/rag/tests/test_ingest.py`

3. Frontend LLM section test:
- `frontend/src/tests/LLMTriggerSections.test.tsx`

## Test Coverage Summary

### Backend

1. `test_llm_trigger_service.py`
- Emotion shift heuristics and chain path
- Process adherence evaluation merge behavior
- Rolling windows and citations

2. `test_interactions_llm_triggers.py`
- API payload mapper shape and field mapping

3. `test_sop_retrieval.py`
- PDF-first SOP retrieval from parsed docs
- Fallback to direct text SOP files
- Priority over Qdrant when manual SOP exists

### RAG

1. `services/rag/tests/test_ingest.py`
- Discovery of PDFs in policy-docs and sop-procedures
- Existing cleaning/chunking/metadata validation

### Frontend

1. `LLMTriggerSections.test.tsx`
- Manager and Agent views render trigger sections correctly

## Operational Runbook

When adding or updating organization docs:

1. Place policy PDFs in `policy-docs`.
2. Place SOP PDFs in `sop-procedures`.
3. Run ingestion to regenerate parsed markdown and vectors.
4. Run `make llm-trigger-test`.
5. Validate in UI interaction detail pages with refresh action.

## Troubleshooting

1. LLM triggers unavailable
- Check backend logs around `evaluate_interaction_triggers`.
- Confirm `include_llm_triggers=true` is passed.

2. SOP steps missing unexpectedly
- Ensure SOP PDFs exist in `sop-procedures`.
- Ensure parsed markdown exists in `sop-standards/{org}/parsed-docs` after ingestion.
- Ensure file stems match expected names.

3. No policy context
- Verify `DOCS_DIR` points to `sop-standards`.
- Re-run ingestion and confirm Qdrant collections are populated.

4. Wrong organization SOP used
- Validate interaction organization mapping.
- Optionally pass explicit `llm_org_filter` to debug.

## Extension Notes

If adding new analysis dimensions:

1. Add schema in `schemas.py`.
2. Add prompt + chain builder.
3. Integrate into `evaluate_interaction_triggers`.
4. Map new fields in interactions route payload mapper.
5. Extend frontend types and views.
6. Add backend + frontend tests and include in `llm-trigger-test` target.
