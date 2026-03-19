# File: backend/app/llm_trigger/schemas.py

## Purpose

Defines strict typed contracts for all LLM Trigger outputs.

## Main Models

- `EmotionShiftAnalysis`
  - `is_dissonance_detected`
  - `dissonance_type`
  - `root_cause`
  - `counterfactual_correction`
- `ProcessAdherenceReport`
  - `detected_topic`
  - `is_resolved`
  - `efficiency_score` (1..10)
  - `missing_sop_steps`
- `NLIEvaluation`
  - `nli_category` with strict Literal values
  - `justification`
- `InteractionLLMTriggerReport`
  - aggregate report for full interaction-level run

## Built-in Guardrails

- `counterfactual_correction` is normalized to start with "If the agent had...".
- `efficiency_score` is range-constrained by Pydantic.
- NLI category is constrained to four accepted values.

## What You Can Modify

- Add optional fields like confidence scores (`float`) for each module.
- Add metadata fields such as `model_name`, `latency_ms`, `trace_id`.
- Tighten `dissonance_type` by converting it from `str` to `Literal` if you want strict allowed labels.

## Risky Changes

- Making required fields optional can weaken downstream validation.
- Renaming fields breaks route responses and client integration.

## What To Tweak First

1. Add confidence fields if you need ranking or thresholding.
2. Convert free-text fields to stricter enums only when prompt reliability is high.
