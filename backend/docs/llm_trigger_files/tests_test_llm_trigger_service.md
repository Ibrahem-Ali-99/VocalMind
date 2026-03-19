# File: backend/tests/test_llm_trigger_service.py

## Purpose

Validates deterministic novelty logic without calling external LLM or vector services.

## Covered Cases

- Non-dissonance path should skip LLM chain.
- Dissonance path should invoke LLM and normalize fallback dissonance type.
- Cross-modal heuristic correctness checks.
- Topic detection and trajectory missing-step helper checks.
- Process adherence merge logic (deterministic + LLM output).

## Test Strategy

- Uses `patch` to mock chain builders.
- Uses fake chain objects to return deterministic schema outputs.
- Keeps tests fast and isolated.

## What You Can Modify

- Add edge-case fixtures for sarcasm, code-switching, and multilingual transcripts.
- Add tests for counterfactual prefix validator behavior.
- Add tests for interaction-level function with mocked DB session responses.

## Risky Changes

- Over-mocking can miss integration failures between route, service, and parser layers.
- Relying only on unit tests may hide real retrieval/LLM timeout behavior.

## What To Tweak First

1. Add one integration-style test for `/interaction/{interaction_id}/run` using patched services.
2. Add regression fixtures from real production-like transcript failures.
