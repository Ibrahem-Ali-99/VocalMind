from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings


logger = logging.getLogger(__name__)

router = APIRouter()

_services_path = str(Path(__file__).resolve().parents[4] / "services")


class RAGQueryRequest(BaseModel):
    query: str
    mode: str = "answer"
    org_filter: str | None = None


class RAGQueryResponse(BaseModel):
    response: str
    chunks: list[dict]
    timing: dict
    retrieval_provenance: list[dict] = Field(default_factory=list)


_engine_lock = threading.Lock()
_engine = None


def _build_retrieval_provenance(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build retrieval provenance cards from chunk metadata.

    This route intentionally stays in the retrieval layer; it surfaces grounded
    evidence and similarity metadata only. The ``verdict`` field is retained
    for response compatibility, but it means retrieval support level, not a
    compliance or NLI judgment.
    """
    retrieval_provenance: list[dict] = []
    supported_threshold = settings.RAG_SUPPORTED_THRESHOLD
    neutral_threshold = settings.RAG_NEUTRAL_THRESHOLD
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        similarity = float(chunk.get("score", 0.0))
        header_path = " > ".join(
            str(metadata.get(key)).strip()
            for key in ("Header 1", "Header 2", "Header 3")
            if metadata.get(key)
        )
        reference = header_path or str(metadata.get("source_file") or metadata.get("doc_id") or "Retrieved chunk")
        retrieval_support_label = (
            "supported"
            if similarity >= supported_threshold
            else "neutral"
            if similarity >= neutral_threshold
            else "insufficient_evidence"
        )
        retrieval_provenance.append(
            {
                "claim": query,
                "chunkRank": chunk.get("rank"),
                "semanticSimilarity": similarity,
                "verdict": retrieval_support_label,
                "docType": metadata.get("doc_type"),
                "policyRef": metadata.get("policy_ref") or [],
                "reference": reference,
                "excerpt": chunk.get("text", "")[:220],
                "provenance": {
                    "docId": metadata.get("doc_id"),
                    "sourceFile": metadata.get("source_file"),
                    "headerPath": header_path or None,
                },
            }
        )
    return retrieval_provenance


@router.get(
    "/health",
    summary="Health check for RAG service dependencies",
)
async def rag_health():
    checks: dict[str, str] = {}
    engine_available = False
    try:
        engine = _get_engine()
        engine_available = True
        collections = [c.name for c in engine.qdrant.get_collections().collections]
        checks["qdrant"] = "ok" if collections else "empty"
    except HTTPException:
        checks["engine"] = "unavailable"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    if engine_available:
        try:
            import httpx as _httpx
            _response = _httpx.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=5.0,
            )
            checks["ollama"] = "ok" if _response.status_code == 200 else f"status:{_response.status_code}"
        except Exception as exc:
            checks["ollama"] = f"error: {exc}"

    checks["groq_api_key"] = "configured" if settings.GROQ_API_KEY else "missing"
    overall = "ok" if all(v == "ok" or v == "configured" for v in checks.values()) else "degraded"
    return {"status": overall, "dependencies": checks}


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            import sys
            if _services_path not in sys.path:
                sys.path.append(_services_path)
            from rag.query_engine import RAGQueryEngine
            _engine = RAGQueryEngine()
        except Exception as exc:
            logger.error("Failed to initialize RAG engine: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG engine: {exc!s}") from exc
        return _engine


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    """Retrieve grounded context from RAG collections and expose provenance."""
    engine = _get_engine()
    try:
        if request.mode == "compliance":
            result = await asyncio.to_thread(engine.query_compliance, text=request.query, org_filter=request.org_filter)
        else:
            result = await asyncio.to_thread(engine.query_answer, question=request.query, org_filter=request.org_filter)
        retrieval_provenance = _build_retrieval_provenance(request.query, result.get("chunks", []))
        return RAGQueryResponse(
            response=result["response"],
            chunks=result.get("chunks", []),
            timing=result.get("timing", {}),
            retrieval_provenance=retrieval_provenance,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
