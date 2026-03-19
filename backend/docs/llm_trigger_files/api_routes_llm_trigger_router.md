# File: backend/app/api/routes/llm_trigger/router.py

## Purpose

FastAPI contract layer for exposing llm_trigger capabilities.

## Endpoints

- `POST /emotion-shift`
- `POST /process-adherence`
- `POST /nli-policy-check`
- `POST /interaction/{interaction_id}/run`

## Request Models

- `EmotionShiftRequest`
- `ProcessAdherenceRequest`
- `NLIPolicyCheckRequest`
- `InteractionTriggerRequest`

## Behavior

- Delegates logic to service functions.
- Converts known service `ValueError` cases to HTTP 404/400 in interaction endpoint.

## What You Can Modify

- Add auth dependencies on each endpoint.
- Add request metadata fields (tenant id, trace id, experiment tag).
- Add endpoint-specific timeout and circuit breaker wrappers.

## Risky Changes

- Changing response models will break frontend or consumer contracts.
- Catching overly broad exceptions can hide real errors and complicate debugging.

## What To Tweak First

1. Add authorization guardrails if this API should be manager-only.
2. Add structured error codes to improve client behavior and observability.
