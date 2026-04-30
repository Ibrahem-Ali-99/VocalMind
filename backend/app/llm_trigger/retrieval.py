"""
Retrieval adapters for the LLM trigger pipeline.

This module belongs to the RAG/retrieval layer from the trigger pipeline's
point of view: it resolves SOP and policy chunks from overrides, manual
fallback files, and Qdrant. It intentionally returns ``RetrievedChunk`` data
only; transcript-level compliance reports and claim-level NLI verdicts live in
their evaluator/checker functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
import httpx
from pathlib import Path
import re
import threading
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "qdrant"
    collection: str | None = None

    @property
    def reference(self) -> str:
        for key in ("source_file", "doc_id", "Header 1", "Header 2", "Header 3"):
            value = self.metadata.get(key)
            if value:
                return str(value)
        return self.source

    @property
    def provenance(self) -> str:
        parts = [
            self.metadata.get("source_file"),
            self.metadata.get("doc_id"),
            " > ".join(
                str(self.metadata.get(key)).strip()
                for key in ("Header 1", "Header 2", "Header 3")
                if self.metadata.get(key)
            ) or None,
        ]
        return " \u2022 ".join(part for part in parts if part)


@dataclass
class ResolvedRetrievalContext:
    text: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    source: str = "unknown"


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z]+", (text or "").lower()) if len(token) > 2}


def _rank_manual_chunks(
    transcript_text: str,
    chunks: list[RetrievedChunk],
    max_chunks: int = 2,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    query_tokens = _tokenize(transcript_text)
    if not query_tokens:
        return chunks[:max_chunks]

    ranked: list[tuple[float, int, RetrievedChunk]] = []
    for index, chunk in enumerate(chunks):
        body_tokens = _tokenize(chunk.text)
        source_tokens = _tokenize(chunk.metadata.get("source_file", ""))
        header_tokens = _tokenize(
            " ".join(
                str(chunk.metadata.get(key, ""))
                for key in ("Header 1", "Header 2", "Header 3")
            )
        )
        overlap = (
            len(query_tokens.intersection(body_tokens))
            + (2 * len(query_tokens.intersection(source_tokens)))
            + len(query_tokens.intersection(header_tokens))
        )
        if chunk.metadata.get("doc_type") in {"sop", "sop-procedures"}:
            overlap += 1
        score = min(1.0, overlap / max(3, len(query_tokens)))
        ranked.append((score, -index, RetrievedChunk(
            text=chunk.text,
            score=score,
            metadata=chunk.metadata,
            source=chunk.source,
            collection=chunk.collection,
        )))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score = ranked[0][0]
    if best_score <= 0:
        return [ranked[0][2]]
    return [chunk for score, _, chunk in ranked if score == best_score][:max_chunks]


def _rank_chunks_by_query(
    query_text: str,
    chunks: list[RetrievedChunk],
    max_chunks: int,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    query_tokens = _tokenize(query_text)
    ranked: list[tuple[float, int, RetrievedChunk]] = []
    for index, chunk in enumerate(chunks):
        overlap = _tokenize(chunk.text)
        lexical_score = 0.0
        if query_tokens and overlap:
            lexical_score = len(query_tokens.intersection(overlap)) / len(query_tokens.union(overlap))
        vector_score = float(chunk.score) if chunk.score is not None else 0.0
        total_score = (vector_score * 0.75) + (lexical_score * 0.25)
        ranked.append((total_score, -index, chunk))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [chunk for _, _, chunk in ranked[:max_chunks]]


_qdrant_lock = threading.Lock()
_shared_qdrant_client: QdrantClient | None = None


def _get_shared_qdrant_client() -> QdrantClient:
    global _shared_qdrant_client
    if _shared_qdrant_client is not None:
        return _shared_qdrant_client
    with _qdrant_lock:
        if _shared_qdrant_client is not None:
            return _shared_qdrant_client
        _shared_qdrant_client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
        return _shared_qdrant_client


class QdrantRetriever:
    """Shared Qdrant retriever for grounded context, with no judgment logic."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        self._qdrant = client or _get_shared_qdrant_client()

    def _embed_query(self, text: str) -> list[float]:
        payloads = (
            ("/api/embed", {"model": settings.EMBEDDING_MODEL, "input": text}),
            ("/api/embeddings", {"model": settings.EMBEDDING_MODEL, "prompt": text}),
        )
        last_error: Exception | None = None
        last_path: str | None = None
        for path, payload in payloads:
            try:
                response = httpx.post(
                    f"{settings.OLLAMA_BASE_URL}{path}",
                    json=payload,
                    timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                vector = data.get("embedding")
                if vector:
                    return vector
            except Exception as exc:
                last_error = exc
                last_path = path
                logger.debug("Embedding endpoint %s failed: %s", path, exc)

        raise ValueError(
            f"Embedding API did not return vector. Last attempt ({last_path}): {last_error}"
        )

    def retrieve_chunks(
        self,
        query_text: str,
        *,
        collection_name: str,
        top_k: int,
        org_filter: str | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embed_query(query_text)

        def _build_filter(include_doc_type: bool) -> Filter | None:
            conditions = []
            if org_filter:
                conditions.append(FieldCondition(key="org", match=MatchValue(value=org_filter)))
            if include_doc_type and doc_type:
                conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
            return Filter(must=conditions) if conditions else None

        def _query(include_doc_type: bool):
            return self._qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=_build_filter(include_doc_type),
            ).points

        points = _query(include_doc_type=bool(doc_type))
        if doc_type and not points:
            logger.warning(
                "No Qdrant chunks matched doc_type=%s in %s; retrying without doc_type filter for legacy data.",
                doc_type,
                collection_name,
            )
            points = _query(include_doc_type=False)

        snippets: list[RetrievedChunk] = []
        for point in points:
            text = ""
            metadata: dict[str, Any] = {}
            if isinstance(point.payload, dict):
                text = str(point.payload.get("text", "")).strip()
                metadata = {str(key): value for key, value in point.payload.items() if key != "text" and value is not None}
                if metadata.get("doc_type") == "sop" and "policy_ref" not in metadata:
                    metadata["policy_ref"] = []
            if text:
                snippets.append(
                    RetrievedChunk(
                        text=text,
                        score=float(point.score) if point.score is not None else None,
                        metadata=metadata,
                        source="qdrant",
                        collection=collection_name,
                    )
                )

        return snippets


class SOPRetriever(QdrantRetriever):
    """Retrieve SOP snippets used as process-adherence context."""

    def retrieve_sop_chunks(self, transcript_text: str, org_filter: str | None = None) -> list[RetrievedChunk]:
        return self.retrieve_chunks(
            query_text=transcript_text,
            collection_name=settings.QDRANT_COLLECTION_SOP_PARENTS,
            top_k=settings.SOP_RETRIEVAL_TOP_K,
            org_filter=org_filter,
            doc_type="sop",
        )

    def retrieve_sop(self, transcript_text: str, org_filter: str | None = None) -> str:
        return "\n\n---\n\n".join(chunk.text for chunk in self.retrieve_sop_chunks(transcript_text, org_filter))


class PolicyRetriever(QdrantRetriever):
    """Retrieve policy chunks used as NLI context and claim provenance."""

    def retrieve_policy_chunks(self, query_text: str, org_filter: str | None = None, top_k: int = 3) -> list[RetrievedChunk]:
        return self.retrieve_chunks(
            query_text=query_text,
            collection_name=settings.QDRANT_COLLECTION_PARENTS,
            top_k=top_k,
            org_filter=org_filter,
            doc_type="policy",
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_root_path(path_value: str) -> Path:
    configured = Path(path_value)
    if configured.is_absolute():
        return configured
    return (_repo_root() / configured).resolve()


def _resolve_sop_docs_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in (settings.SOP_DOCS_ROOT, settings.KNOWLEDGE_DOCS_ROOT):
        resolved = _resolve_root_path(candidate)
        if resolved in roots:
            continue
        roots.append(resolved)
    return roots


def _resolve_sop_parsed_docs_root() -> Path:
    return _resolve_root_path(settings.SOP_PARSED_DOCS_ROOT)


def _resolve_org_parsed_docs_dir(org_filter: str) -> Path:
    parsed_root = _resolve_sop_parsed_docs_root()
    return parsed_root / org_filter / "parsed-docs" / "sops"


def _read_manual_org_sop_chunks(org_filter: str | None) -> list[RetrievedChunk]:
    if not org_filter:
        return []

    parsed_docs_root = _resolve_org_parsed_docs_dir(org_filter)
    sop_dirs: list[Path] = []
    for docs_root in _resolve_sop_docs_roots():
        for folder in ("sop-procedures", "faq-docs"):
            candidate = docs_root / org_filter / folder
            if not candidate.exists() or not candidate.is_dir():
                continue
            if candidate in sop_dirs:
                continue
            sop_dirs.append(candidate)

    if not sop_dirs:
        return []

    pdf_paths: list[Path] = []
    seen_pdf_paths: set[Path] = set()
    for sop_dir in sop_dirs:
        for path in sorted(path for path in sop_dir.rglob("*.pdf") if path.is_file()):
            resolved = path.resolve()
            if resolved in seen_pdf_paths:
                continue
            seen_pdf_paths.add(resolved)
            pdf_paths.append(path)

    chunks: list[RetrievedChunk] = []

    for pdf_path in pdf_paths:
        converted_md = parsed_docs_root / f"{pdf_path.stem}.md"
        if not converted_md.exists() or not converted_md.is_file():
            converted_md = parsed_docs_root.parent / f"{pdf_path.stem}.md"
        if not converted_md.exists() or not converted_md.is_file():
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
        chunks.append(
            RetrievedChunk(
                text=f"[{pdf_path.name}]\n{text}",
                score=None,
                metadata={
                    "source_file": pdf_path.name,
                    "doc_type": "sop",
                    "policy_ref": [],
                },
                source="manual",
            )
        )

    if chunks:
        return chunks

    allowed_suffixes = {".md", ".txt"}

    for sop_dir in sop_dirs:
        for path in sorted(sop_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if not text:
                continue
            chunks.append(
                RetrievedChunk(
                    text=f"[{path.name}]\n{text}",
                    score=None,
                    metadata={
                        "source_file": path.name,
                        "doc_type": "sop",
                        "policy_ref": [],
                    },
                    source="manual",
                )
            )

    return chunks


def _join_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(chunk.text for chunk in chunks if chunk.text)


def _read_manual_org_sop_docs(org_filter: str | None) -> str:
    return _join_chunks(_read_manual_org_sop_chunks(org_filter))


def resolve_retrieved_sop_context(
    transcript_text: str,
    retrieved_sop_from_pinecone: str | None,
    org_filter: str | None = None,
    max_chunks: int = 2,
) -> ResolvedRetrievalContext:
    """
    Resolve SOP grounding context without scoring process adherence.

    Precedence is preserved for compatibility:
      1) caller-provided override,
      2) org-local manual SOP files,
      3) Qdrant SOP parent collection.
    """
    if retrieved_sop_from_pinecone and retrieved_sop_from_pinecone.strip():
        chunk = RetrievedChunk(
            text=retrieved_sop_from_pinecone.strip(),
            score=None,
            metadata={"source": "override"},
            source="override",
        )
        return ResolvedRetrievalContext(text=chunk.text, chunks=[chunk], source="override")

    manual_chunks = _read_manual_org_sop_chunks(org_filter)
    if manual_chunks:
        ranked_manual_chunks = _rank_manual_chunks(
            transcript_text=transcript_text,
            chunks=manual_chunks,
            max_chunks=max_chunks,
        )
        return ResolvedRetrievalContext(
            text=_join_chunks(ranked_manual_chunks),
            chunks=ranked_manual_chunks,
            source="manual",
        )

    qdrant_chunks: list[RetrievedChunk] = []
    try:
        retriever = SOPRetriever()
        qdrant_chunks = retriever.retrieve_sop_chunks(transcript_text=transcript_text, org_filter=org_filter)
    except Exception:
        qdrant_chunks = []

    if qdrant_chunks:
        ranked_qdrant_chunks = _rank_chunks_by_query(
            query_text=transcript_text,
            chunks=qdrant_chunks,
            max_chunks=max_chunks,
        )
        return ResolvedRetrievalContext(
            text=_join_chunks(ranked_qdrant_chunks),
            chunks=ranked_qdrant_chunks,
            source="qdrant",
        )

    return ResolvedRetrievalContext(text="", chunks=[], source="none")


def retrieve_policy_chunks(
    query_text: str,
    org_filter: str | None = None,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    retriever = PolicyRetriever()
    return retriever.retrieve_policy_chunks(query_text=query_text, org_filter=org_filter, top_k=top_k)


def resolve_retrieved_sop(
    transcript_text: str,
    retrieved_sop_from_pinecone: str | None,
    org_filter: str | None = None,
) -> str:
    return resolve_retrieved_sop_context(
        transcript_text=transcript_text,
        retrieved_sop_from_pinecone=retrieved_sop_from_pinecone,
        org_filter=org_filter,
    ).text
