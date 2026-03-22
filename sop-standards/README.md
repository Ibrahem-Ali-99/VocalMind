# SOP Standards & Policy Documentation

Centralized storage for organization-specific policies and standard operating procedures.

## Directory Structure

```
sop-standards/
└── {organization}/
    ├── policy-docs/         ← Policy PDFs (input for RAG ingestion)
    │   ├── refund-policy.pdf
    │   ├── billing-guidelines.pdf
    │   └── compliance-requirements.pdf
    │
    └── sop-procedures/      ← SOP source PDFs (converted by Docling)
        ├── 01-refund-request-processing.pdf
        ├── 02-billing-issue-resolution.pdf
        ├── 03-technical-support-troubleshooting.pdf
        └── 04-account-access-recovery.pdf
```

## Organization: Nexalink

### Policy Docs (`policy-docs/`)
- **Purpose**: Input for RAG service document ingestion
- **Format**: PDF files (auto-converted to Markdown via Docling)
- **Usage**: Provides context for SOP generation and Qdrant vector search
- **Ingestion**: Run `services/rag/ingest.py` to process

### SOP Procedures (`sop-procedures/`)
- **Purpose**: Manual SOP documents for agent guidance
- **Format**: PDF files (converted to Markdown via Docling during ingestion)
- **Usage**: Backend reads converted Markdown from `services/rag/parsed_docs`
- **Priority**: Checked before Qdrant fallback in `backend/app/llm_trigger/retrieval.py`

## Workflow

### 1. Create Policy Docs
- Place your policy PDFs in `sop-standards/{org}/policy-docs/`
- These are reference documents for the RAG system

### 2. Generate SOPs
- Use the SOP generation prompt with your policy PDFs as input
- Output: PDF SOP documents

### 3. Place Generated SOPs
- Save generated `.pdf` files to `sop-standards/{org}/sop-procedures/`
- Naming: Sequential (`01-*.pdf`, `02-*.pdf`, etc.)
- Run ingestion so Docling creates Markdown in `services/rag/parsed_docs`

### 4. Ingest to Qdrant (Optional Fallback)
```bash
cd services/rag
python -c "from ingest import DocumentIngestionPipeline; p = DocumentIngestionPipeline(); p.run()"
```
- Converts PDFs → Markdown via Docling
- Ingests both `policy-docs/*.pdf` and `sop-procedures/*.pdf`
- Stores in Qdrant for vector search fallback

## System Integration

- **LLM Trigger Service**: Checks `sop-procedures/` first, then Qdrant
- **Frontend**: Displays SOP adherence metrics in manager/agent views
- **Config**: Backend uses `SOP_DOCS_ROOT: str = "sop-standards"` in `backend/app/core/config.py`

## Adding a New Organization

1. Create folder: `sop-standards/{new_org}/`
2. Create subfolders:
   - `policy-docs/` → Place PDFs here
    - `sop-procedures/` → Place SOP PDFs here
3. Generate SOPs using the prompt with your policies
4. Run ingestion to convert PDFs to Markdown and index vectors

---

**Last Updated**: 2026-03-21
