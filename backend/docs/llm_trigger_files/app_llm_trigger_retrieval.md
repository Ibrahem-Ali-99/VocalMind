# File: backend/app/llm_trigger/retrieval.py

## Purpose

Retrieves SOP context from Qdrant using embeddings, with optional organization filtering.

## Main Components

- `SOPRetriever`
  - `_embed_query(text)` calls Ollama embeddings endpoint
  - `retrieve_sop(transcript_text, org_filter)` queries Qdrant parents collection
- `resolve_retrieved_sop(...)`
  - uses provided SOP if available
  - otherwise falls back to Qdrant retrieval

## What You Can Modify

- Swap embedding backend from Ollama to external APIs.
- Adjust `SOP_RETRIEVAL_TOP_K` for more/less context.
- Add score thresholding to remove weakly relevant chunks.

## Risky Changes

- Returning too many chunks can hurt LLM focus and increase token cost.
- No timeout/error handling at caller level can fail entire pipeline when vector infra is down.

## What To Tweak First

1. Calibrate top-k per topic domain.
2. Add chunk metadata (policy id/title) to retrieval output if you want stronger attribution in reports.
