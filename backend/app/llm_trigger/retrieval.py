from __future__ import annotations

import httpx
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
            collection_name=settings.QDRANT_COLLECTION_PARENTS,
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


def resolve_retrieved_sop(
    transcript_text: str,
    retrieved_sop_from_pinecone: str | None,
    org_filter: str | None = None,
) -> str:
    if retrieved_sop_from_pinecone and retrieved_sop_from_pinecone.strip():
        return retrieved_sop_from_pinecone.strip()

    retriever = SOPRetriever()
    return retriever.retrieve_sop(transcript_text=transcript_text, org_filter=org_filter)
