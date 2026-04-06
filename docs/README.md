# VocalMind Documentation Index

This folder contains the team-facing technical documentation for the main evaluation and retrieval subsystems in VocalMind.

## Start Here

1. [Evidence-Anchored Explainability Layer](./explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md)
- Unified guide for span-level trigger attribution, claim provenance, API fields, and manager Evidence Card behavior

2. [LLM Trigger Feature Guide](./llm_trigger/LLM_TRIGGER_FEATURE_GUIDE.md)
- Architecture, SOP retrieval flow, payload behavior, testing, and maintenance notes for LLM-trigger evaluation

3. [RAG Overview](./rag/RAG_OVERVIEW.md)
- Retrieval architecture, dual-collection indexing, provenance flow, and runtime components

4. [RAG Ingestion Pipeline](./rag/INGESTION_PIPELINE.md)
- Document ingestion, parsing, chunking, and indexing workflow for policy and SOP content

## Documentation Areas

### `explainability/`

Cross-cutting documentation for the shared explainability layer that links:

1. transcript spans
2. retrieved SOP/policy evidence
3. verdicts shown in the manager UI

### `llm_trigger/`

Documentation for:

1. emotion-trigger analysis
2. SOP/process-adherence reasoning
3. NLI policy review
4. interaction-detail explainability payload mapping

### `rag/`

Documentation for:

1. document ingestion
2. retrieval and synthesis
3. compliance and answer-evaluation flows
4. retrieval provenance surfaced to explainability consumers

## Suggested Reading Paths

### If you are new to the project

1. Read [Evidence-Anchored Explainability Layer](./explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md)
2. Read [LLM Trigger Feature Guide](./llm_trigger/LLM_TRIGGER_FEATURE_GUIDE.md)
3. Read [RAG Overview](./rag/RAG_OVERVIEW.md)

### If you are debugging manager call review

1. Read [Evidence-Anchored Explainability Layer](./explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md)
2. Read [LLM Trigger Feature Guide](./llm_trigger/LLM_TRIGGER_FEATURE_GUIDE.md)

### If you are debugging policy retrieval or provenance

1. Read [RAG Overview](./rag/RAG_OVERVIEW.md)
2. Read [RAG Ingestion Pipeline](./rag/INGESTION_PIPELINE.md)
3. Read [Evidence-Anchored Explainability Layer](./explainability/EVIDENCE_ANCHORED_EXPLAINABILITY_LAYER.md)
