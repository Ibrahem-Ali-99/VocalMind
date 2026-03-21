# LLM Trigger Documentation

This folder is intentionally minimal.

## What is kept here

1. `LLM_TRIGGER_FEATURE_GUIDE.md`
	- Single comprehensive guide for architecture, data flow, SOP/PDF flow, API behavior, testing, and troubleshooting.

2. `README.md`
	- This index and usage note.

## Test source of truth

Tests are maintained in code, not as separate per-function markdown docs.

1. `backend/tests/test_llm_trigger_service.py`
2. `backend/tests/test_interactions_llm_triggers.py`
3. `backend/tests/test_sop_retrieval.py`
4. `services/rag/tests/test_ingest.py`
5. `frontend/src/tests/LLMTriggerSections.test.tsx`

## How to run validation

Preferred (if `make` is installed):

`make llm-trigger-test`
