# File: backend/app/llm_trigger/service.py

## Purpose

Core orchestration layer for all LLM Trigger logic.

## Responsibilities

- Calls LCEL chains.
- Applies deterministic pre/post logic.
- Reconstructs transcript context from DB models.
- Runs full interaction-level evaluation in parallel.

## Key Public Functions

- `analyze_emotion_shift(...)`
- `evaluate_process_adherence(...)`
- `run_nli_policy_check(...)`
- `evaluate_interaction_triggers(...)`

## Deterministic Logic Added

- Cross-modal dissonance gate:
  - `_detect_cross_modal_dissonance(...)`
  - Uses text polarity + acoustic polarity mismatch rule.
- Topic and process trajectory mapping:
  - `_detect_topic(...)`
  - `RESOLUTION_GRAPHS`
  - `_trajectory_missing_steps(...)`
  - `_efficiency_score_heuristic(...)`
  - `_is_resolved_heuristic(...)`

## Input Derivation Rules

- Customer text: first customer utterance with text.
- Acoustic emotion: emotion of that customer utterance (fallback neutral).
- Agent statement: latest seen agent utterance text.
- Transcript text: `Transcript.full_text` else reconstructed utterance lines.

## What You Can Modify

- Expand sentiment lexicons for domain-specific language.
- Replace lexical polarity with trained sentiment model.
- Replace static `RESOLUTION_GRAPHS` with DB-managed topic graphs.
- Add confidence blending between deterministic and LLM components.

## Risky Changes

- Overly aggressive dissonance gating may skip valid sarcasm cases.
- Weak heuristics can dominate LLM output if merged too strongly.
- Topic misclassification cascades into wrong resolution graph and unfair scoring.

## What To Tweak First

1. Tune polarity lexicons using real transcripts.
2. Improve topic detector with retrieval metadata + keyword weighting.
3. Version your graph templates and evaluate metrics per version.
