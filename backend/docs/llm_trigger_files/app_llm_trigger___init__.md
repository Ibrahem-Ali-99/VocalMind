# File: backend/app/llm_trigger/__init__.py

## Purpose

Provides package-level exports so other modules can import core LLM Trigger types and functions from one place.

## What It Exposes

- Schemas:
  - `EmotionShiftAnalysis`
  - `ProcessAdherenceReport`
  - `NLIEvaluation`
  - `InteractionLLMTriggerReport`
- Service functions:
  - `analyze_emotion_shift`
  - `evaluate_process_adherence`
  - `run_nli_policy_check`

## What You Can Modify

- Add exports for new helper functions if they become part of the public API.
- Keep `__all__` aligned with imports to avoid stale exports.

## Risky Changes

- Removing symbols from `__all__` can break imports in other files.
- Renaming exported symbols without updating call sites will break runtime imports.

## What To Tweak First

- Keep this file minimal and stable.
- Treat this as your public package interface contract.
