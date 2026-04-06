from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import importlib.util
import os
import sys


# Import services module which is parallel to backend.
services_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../services"))
if services_path not in sys.path:
    sys.path.append(services_path)

router = APIRouter()


class RAGQueryRequest(BaseModel):
    query: str
    mode: str = "answer"
    org_filter: str | None = None


class RAGQueryResponse(BaseModel):
    response: str
    chunks: list[dict]
    timing: dict
    retrieval_provenance: list[dict] = Field(default_factory=list)


# Lazy initialize engine
_engine = None


def get_engine():
    global _engine

    if importlib.util.find_spec("rag.query_engine") is None:
        raise HTTPException(status_code=503, detail="RAG service is not available (imports failed or not installed).")

    if _engine is None:
        try:
            from rag.query_engine import RAGQueryEngine

            _engine = RAGQueryEngine()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG engine: {exc!s}")

    return _engine


@router.post("/query", response_model=RAGQueryResponse)
def query_rag_endpoint(request: RAGQueryRequest):
    engine = get_engine()
    try:
        if request.mode == "compliance":
            result = engine.query_compliance(text=request.query, org_filter=request.org_filter)
        else:
            result = engine.query_answer(question=request.query, org_filter=request.org_filter)
        retrieval_provenance: list[dict] = []
        for chunk in result.get("chunks", []):
            metadata = chunk.get("metadata", {})
            similarity = float(chunk.get("score", 0.0))
            header_path = " > ".join(
                str(metadata.get(key)).strip()
                for key in ("Header 1", "Header 2", "Header 3")
                if metadata.get(key)
            )
            reference = header_path or str(metadata.get("source_file") or metadata.get("doc_id") or "Retrieved chunk")
            verdict = "supported" if similarity >= 0.82 else "neutral" if similarity >= 0.55 else "insufficient_evidence"
            retrieval_provenance.append(
                {
                    "claim": request.query,
                    "chunkRank": chunk.get("rank"),
                    "semanticSimilarity": similarity,
                    "verdict": verdict,
                    "reference": reference,
                    "excerpt": chunk.get("text", "")[:220],
                    "provenance": {
                        "docId": metadata.get("doc_id"),
                        "sourceFile": metadata.get("source_file"),
                        "headerPath": header_path or None,
                    },
                }
            )
        return RAGQueryResponse(
            response=result["response"],
            chunks=result.get("chunks", []),
            timing=result.get("timing", {}),
            retrieval_provenance=retrieval_provenance,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
