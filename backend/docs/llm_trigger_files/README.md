# LLM Trigger File Guide

This folder documents each newly created file for the LLM Trigger feature.

## Files Covered

1. `backend/app/llm_trigger/__init__.py`
2. `backend/app/llm_trigger/schemas.py`
3. `backend/app/llm_trigger/prompts.py`
4. `backend/app/llm_trigger/chains.py`
5. `backend/app/llm_trigger/retrieval.py`
6. `backend/app/llm_trigger/service.py`
7. `backend/app/api/routes/llm_trigger/__init__.py`
8. `backend/app/api/routes/llm_trigger/router.py`
9. `backend/tests/test_llm_trigger_service.py`

## Supporting Edited Files (not newly created)

- `backend/app/api/main.py`
- `backend/app/core/config.py`
- `backend/pyproject.toml`
- `.gitignore`

## How To Use This Guide

- Open the matching markdown file for a code file.
- Read "What to tweak first" for practical tuning knobs.
- Read "Risky changes" before altering interfaces or schema contracts.
