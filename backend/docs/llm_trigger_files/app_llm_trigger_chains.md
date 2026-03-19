# File: backend/app/llm_trigger/chains.py

## Purpose

Builds LCEL chains in the required pattern:

`prompt | model | parser`

## Main Functions

- `_build_model()`
- `build_emotion_shift_chain()`
- `build_process_adherence_chain()`
- `build_nli_policy_chain()`

## Dependencies

- `ChatGroq` from `langchain_groq`
- `PydanticOutputParser`
- settings from `app.core.config`

## What You Can Modify

- Tune model parameters:
  - `LLM_MODEL`
  - `LLM_TEMPERATURE`
  - `LLM_MAX_TOKENS`
- Inject a fallback model provider behind an interface if needed.
- Add retry wrappers around `ainvoke` at service layer.

## Risky Changes

- Changing parser schema without matching prompt constraints can break structured parsing.
- High temperature may reduce deterministic grading quality.

## What To Tweak First

1. Keep temperature at `0.0` or very low for evaluator stability.
2. Increase max tokens only if truncation is observed.
