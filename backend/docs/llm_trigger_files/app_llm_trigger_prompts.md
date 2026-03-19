# File: backend/app/llm_trigger/prompts.py

## Purpose

Holds all LangChain prompt templates and few-shot examples for the three features.

## Prompt Builders

- `build_emotion_shift_prompt()`
- `build_process_adherence_prompt()`
- `build_nli_policy_prompt()`

## Few-Shot Blocks

- `EMOTION_SHIFT_FEW_SHOT`
- `NLI_FEW_SHOT`

## Design Notes

- Prompts enforce JSON output via parser format instructions.
- Process adherence prompt includes:
  - `topic_hint`
  - `retrieved_sop`
  - `expected_resolution_graph`

## What You Can Modify

- Add domain-specific examples (telecom, banking, ecommerce).
- Make NLI examples more adversarial to reduce hallucination errors.
- Add policy section identifiers in prompt context for better grounded references.

## Risky Changes

- Overly long few-shot examples increase token cost and latency.
- Ambiguous wording can reduce parser success and increase invalid JSON outputs.

## What To Tweak First

1. Expand few-shot examples with your real call-center edge cases.
2. Keep instructions short but explicit to preserve structure quality.
