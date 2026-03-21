from __future__ import annotations

import httpx
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings


class SOPRetriever:
    """Minimal Qdrant retriever for SOP snippets used by llm_trigger chains."""

    def __init__(self) -> None:
        self._qdrant = QdrantClient(url=settings.QDRANT_URL)

    def _embed_query(self, text: str) -> list[float]:
        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/embeddings",
            json={"model": settings.EMBEDDING_MODEL, "prompt": text},
            timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        vector = payload.get("embedding")
        if not vector:
            raise ValueError("Embedding API did not return vector.")
        return vector

    def retrieve_sop(self, transcript_text: str, org_filter: str | None = None) -> str:
        query_vector = self._embed_query(transcript_text)

        query_filter = None
        if org_filter:
            query_filter = Filter(
                must=[FieldCondition(key="org", match=MatchValue(value=org_filter))]
            )

        points = self._qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_SOP_PARENTS,
            query=query_vector,
            limit=settings.SOP_RETRIEVAL_TOP_K,
            query_filter=query_filter,
        ).points

        snippets: list[str] = []
        for point in points:
            text = ""
            if isinstance(point.payload, dict):
                text = str(point.payload.get("text", "")).strip()
            if text:
                snippets.append(text)

        return "\n\n---\n\n".join(snippets)


def _repo_root() -> Path:
    # backend/app/llm_trigger/retrieval.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _resolve_sop_docs_root() -> Path:
    configured = Path(settings.SOP_DOCS_ROOT)
    if configured.is_absolute():
        return configured
    return (_repo_root() / configured).resolve()


def _resolve_sop_parsed_docs_root() -> Path:
    configured = Path(settings.SOP_PARSED_DOCS_ROOT)
    if configured.is_absolute():
        return configured
    return (_repo_root() / configured).resolve()


def _resolve_org_parsed_docs_dir(org_filter: str) -> Path:
    parsed_root = _resolve_sop_parsed_docs_root()
    return parsed_root / org_filter / "parsed-docs"


def _read_manual_org_sop_docs(org_filter: str | None) -> str:
    if not org_filter:
        return ""

    docs_root = _resolve_sop_docs_root()
    sop_dir = docs_root / org_filter / "sop-procedures"
    if not sop_dir.exists() or not sop_dir.is_dir():
        return ""

    # SOP source of truth is PDF. We read Docling-converted markdown from
    # sop-standards/{org}/parsed-docs by matching PDF basenames.
    parsed_docs_root = _resolve_org_parsed_docs_dir(org_filter)
    pdf_paths = sorted(path for path in sop_dir.rglob("*.pdf") if path.is_file())
    chunks: list[str] = []

    for pdf_path in pdf_paths:
        converted_md = parsed_docs_root / f"{pdf_path.stem}.md"
        if not converted_md.exists() or not converted_md.is_file():
            # Backward compatibility: old shared parsed folder layout.
            legacy_converted_md = _repo_root() / "services" / "rag" / "parsed_docs" / f"{pdf_path.stem}.md"
            converted_md = legacy_converted_md
        if not converted_md.exists() or not converted_md.is_file():
            continue
        try:
            text = converted_md.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text:
            continue
        chunks.append(f"[{pdf_path.name}]\n{text}")

    if chunks:
        return "\n\n---\n\n".join(chunks)

    # Backward compatibility: allow direct text SOP files if present.
    allowed_suffixes = {".md", ".txt"}

    for path in sorted(sop_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text:
            continue
        chunks.append(f"[{path.name}]\n{text}")

    return "\n\n---\n\n".join(chunks)


def resolve_retrieved_sop(
    transcript_text: str,
    retrieved_sop_from_pinecone: str | None,
    org_filter: str | None = None,
) -> str:
    if retrieved_sop_from_pinecone and retrieved_sop_from_pinecone.strip():
        return retrieved_sop_from_pinecone.strip()

    manual_sop = _read_manual_org_sop_docs(org_filter)
    if manual_sop:
        return manual_sop

    retriever = SOPRetriever()
    return retriever.retrieve_sop(transcript_text=transcript_text, org_filter=org_filter)
