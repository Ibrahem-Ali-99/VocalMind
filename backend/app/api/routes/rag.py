import os
import sys

# Import services module which is parallel to backend
services_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../services"))
if services_path not in sys.path:
    sys.path.append(services_path)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
try:
    from rag.query_engine import RAGQueryEngine
    engine_available = True
except ImportError:
    engine_available = False

router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str
    mode: str = "answer"
    org_filter: str | None = None

class RAGQueryResponse(BaseModel):
    response: str
    chunks: list[dict]
    timing: dict

# Lazy initialize engine
_engine = None

def get_engine():
    global _engine
    if not engine_available:
        raise HTTPException(status_code=503, detail="RAG service is not available (imports failed or not installed).")
    if _engine is None:
        try:
            from rag.query_engine import RAGQueryEngine
            _engine = RAGQueryEngine()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG engine: {str(e)}")
    return _engine

@router.post("/query", response_model=RAGQueryResponse)
def query_rag_endpoint(request: RAGQueryRequest):
    engine = get_engine()
    try:
        if request.mode == "compliance":
            result = engine.query_compliance(text=request.query, org_filter=request.org_filter)
        else:
            result = engine.query_answer(question=request.query, org_filter=request.org_filter)
        return RAGQueryResponse(
            response=result["response"],
            chunks=result.get("chunks", []),
            timing=result.get("timing", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
