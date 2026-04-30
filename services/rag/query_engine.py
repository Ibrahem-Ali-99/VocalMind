"""
VocalMind Final RAG — Retrieval + optional synthesis engine.

This module is the RAG retrieval layer for policy/SOP grounding.
It owns:
  1) embedding queries,
  2) vector retrieval from Qdrant collections,
  3) structured retrieval payload formatting.

It may optionally synthesize a response for compatibility endpoints, but it does
not perform compliance judging or NLI verdicting. Judging belongs to:
  - services/rag/evaluator.py (transcript-level policy compliance reports)
  - backend/app/llm_trigger/service.py (single-claim NLI policy checks)
"""

import json
import logging
import time
from datetime import datetime

import httpx
from groq import Groq
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.groq import Groq as LlamaGroq
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

try:
    from .config import settings
except ImportError:  # pragma: no cover - allows direct script/test imports
    from config import settings


logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """
    RAG retrieval engine backed by Qdrant vector search.

    Retrieval-first APIs:
      - ``retrieve_policy_context(text)``  → parent chunks (policy context)
      - ``retrieve_answer_context(question)`` → child chunks (answer context)
      - ``retrieve_context(question, collection)`` → generic retrieval

    Compatibility APIs with synthesis:
      - ``query_compliance(text)`` and ``query_answer(question)``
      - ``query(question, collection)``
    """

    def __init__(self) -> None:
        self.qdrant = QdrantClient(url=settings.qdrant.url)
        self._setup_llm()
        self.logs_dir = settings.BASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def _setup_llm(self) -> None:
        """Configure Groq LLM for response synthesis via LlamaIndex."""
        self.llm = LlamaGroq(
            model=settings.groq.model,
            api_key=settings.groq.api_key.get_secret_value(),
            temperature=settings.groq.temperature,
            max_tokens=settings.groq.max_tokens,
            context_window=settings.groq.context_window,
        )
        self.synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=settings.response_mode,
        )
        # Also keep a raw Groq client for structured prompts (evaluator uses it)
        self.groq_client = Groq(api_key=settings.groq.api_key.get_secret_value())

    # ── Embedding ─────────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query string via Ollama."""
        retry_delays = (0.4, 1.0, 2.0)
        payloads = (
            ("/api/embed", {"model": settings.embedding.model, "input": text}),
            ("/api/embeddings", {"model": settings.embedding.model, "prompt": text}),
        )
        last_error: Exception | None = None
        last_path: str | None = None
        for delay in (0.0, *retry_delays):
            if delay:
                time.sleep(delay)

            for path, payload in payloads:
                try:
                    response = httpx.post(
                        f"{settings.embedding.base_url}{path}",
                        json=payload,
                        timeout=settings.embedding.request_timeout,
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

        raise ConnectionError(
            f"Cannot reach Ollama embeddings API at {settings.embedding.base_url} "
            f"(last attempted endpoint: {last_path}): {last_error}"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_text: str,
        collection: str,
        top_k: int | None = None,
        org_filter: str | None = None,
        doc_type: str | None = None,
    ) -> tuple[list[ScoredPoint], float]:
        """
        Embed the query and search Qdrant.

        Returns:
            (scored_points, retrieval_seconds)
        """
        top_k = top_k or settings.similarity_top_k

        t0 = time.perf_counter()
        query_vector = self._embed_query(query_text)

        def _build_filter(include_doc_type: bool) -> Filter | None:
            conditions = []
            if org_filter:
                conditions.append(FieldCondition(key="org", match=MatchValue(value=org_filter)))
            if include_doc_type and doc_type:
                conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
            return Filter(must=conditions) if conditions else None

        query_filter = _build_filter(include_doc_type=bool(doc_type))
        results = self.qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        if doc_type and not results:
            logger.warning(
                "No Qdrant chunks matched doc_type=%s in %s; retrying without doc_type filter for legacy data.",
                doc_type,
                collection,
            )
            results = self.qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                query_filter=_build_filter(include_doc_type=False),
            ).points
        retrieval_time = time.perf_counter() - t0
        return results, retrieval_time

    @staticmethod
    def _scored_points_to_nodes(points: list[ScoredPoint]) -> list[NodeWithScore]:
        """Convert Qdrant ScoredPoint results to LlamaIndex NodeWithScore."""
        nodes: list[NodeWithScore] = []
        for pt in points:
            text = pt.payload.get("text", "")
            metadata = {k: v for k, v in pt.payload.items() if k != "text"}
            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=pt.score))
        return nodes

    # ── Synthesis ─────────────────────────────────────────────────────────

    def _synthesize(
        self,
        question: str,
        nodes: list[NodeWithScore],
    ) -> tuple[str, float]:
        """Run LlamaIndex response synthesis and return (response_text, seconds)."""
        t0 = time.perf_counter()
        response = self.synthesizer.synthesize(question, nodes=nodes)
        synthesis_time = time.perf_counter() - t0
        return str(response), synthesis_time

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_query(
        self,
        question: str,
        collection: str,
        chunks: list[dict],
        response_text: str,
        timing: dict,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"query_{timestamp}.json"
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "collection": collection,
            "model": settings.groq.model,
            "similarity_top_k": settings.similarity_top_k,
            "timing_seconds": timing,
            "retrieved_chunks": chunks,
            "response": response_text,
        }
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: Failed to log query: {e}")

    # ── Public API ────────────────────────────────────────────────────────

    @staticmethod
    def _format_retrieved_chunks(scored_points: list[ScoredPoint]) -> list[dict]:
        chunks: list[dict] = []
        for i, pt in enumerate(scored_points, 1):
            chunks.append(
                {
                    "rank": i,
                    "score": float(pt.score),
                    "metadata": {k: v for k, v in pt.payload.items() if k != "text"},
                    "text": pt.payload.get("text", ""),
                    "text_length": len(pt.payload.get("text", "")),
                }
            )
        return chunks

    def retrieve_context(
        self,
        question: str,
        collection: str | None = None,
        top_k: int | None = None,
        org_filter: str | None = None,
        doc_type: str | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Retrieve grounded context chunks only (no judgment, no verdicting).

        Returns a retrieval payload shaped like the query payload for backward
        compatibility: ``{"response": "", "chunks": [...], "timing": {...}}``.
        """
        collection = collection or settings.qdrant.collection_children
        scored_points, retrieval_time = self._retrieve(question, collection, top_k, org_filter, doc_type)
        chunks = self._format_retrieved_chunks(scored_points)

        if verbose and chunks:
            print(f"\n{'='*60}")
            print(f"RETRIEVED CHUNKS ({retrieval_time:.2f}s) from [{collection}]")
            print(f"{'='*60}")
            for c in chunks:
                print(f"\n  [{c['rank']}] Score: {c['score']:.4f}")
                meta_str = " | ".join(
                    f"{k}: {v}" for k, v in c["metadata"].items() if k not in ("text", "ingested_at")
                )
                print(f"      Meta: {meta_str}")
                print(f"      Preview: {c['text'][:150]}...")

        timing = {
            "retrieval": round(retrieval_time, 4),
            "synthesis": 0.0,
            "total": round(retrieval_time, 4),
        }
        self._log_query(question, collection, chunks, "", timing)
        return {"response": "", "chunks": chunks, "timing": timing}

    def retrieve_policy_context(
        self,
        text: str,
        org_filter: str | None = None,
        top_k: int | None = None,
        verbose: bool = False,
    ) -> dict:
        """Retrieve policy-grounding context from the parents collection."""
        return self.retrieve_context(
            question=text,
            collection=settings.qdrant.collection_parents,
            top_k=top_k,
            org_filter=org_filter,
            doc_type="policy",
            verbose=verbose,
        )

    def retrieve_answer_context(
        self,
        question: str,
        org_filter: str | None = None,
        top_k: int | None = None,
        doc_type: str | None = None,
        verbose: bool = False,
    ) -> dict:
        """Retrieve answer-grounding context from the children collection."""
        return self.retrieve_context(
            question=question,
            collection=settings.qdrant.collection_children,
            top_k=top_k,
            org_filter=org_filter,
            doc_type=doc_type,
            verbose=verbose,
        )

    def query(
        self,
        question: str,
        collection: str | None = None,
        top_k: int | None = None,
        org_filter: str | None = None,
        doc_type: str | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Execute a RAG query against a Qdrant collection.

        Args:
            question:    The query text.
            collection:  Qdrant collection name. Defaults to children.
            top_k:       Number of results. Defaults to config.similarity_top_k.
            org_filter:  Optional org name to filter results.
            verbose:     Print retrieved chunks and timing.

        Returns:
            dict with keys: response, chunks, timing
        """
        collection = collection or settings.qdrant.collection_children

        # 1. Retrieve
        scored_points, retrieval_time = self._retrieve(question, collection, top_k, org_filter, doc_type)
        nodes = self._scored_points_to_nodes(scored_points)

        # 2. Format chunks for display/logging
        chunks = self._format_retrieved_chunks(scored_points)

        if verbose and chunks:
            print(f"\n{'='*60}")
            print(f"RETRIEVED CHUNKS ({retrieval_time:.2f}s) from [{collection}]")
            print(f"{'='*60}")
            for c in chunks:
                print(f"\n  [{c['rank']}] Score: {c['score']:.4f}")
                meta_str = " | ".join(
                    f"{k}: {v}" for k, v in c["metadata"].items()
                    if k not in ("text", "ingested_at")
                )
                print(f"      Meta: {meta_str}")
                print(f"      Preview: {c['text'][:150]}...")

        # 3. Synthesise
        if not nodes:
            response_text = "No relevant documents found."
            synthesis_time = 0.0
        else:
            response_text, synthesis_time = self._synthesize(question, nodes)

        total_time = retrieval_time + synthesis_time

        if verbose:
            print(
                f"\n  Retrieval {retrieval_time:.2f}s | "
                f"Synthesis {synthesis_time:.2f}s | "
                f"Total {total_time:.2f}s"
            )

        timing = {
            "retrieval": round(retrieval_time, 4),
            "synthesis": round(synthesis_time, 4),
            "total": round(total_time, 4),
        }

        # 4. Log
        self._log_query(question, collection, chunks, response_text, timing)

        return {
            "response": response_text,
            "chunks": chunks,
            "timing": timing,
        }

    def query_compliance(
        self, text: str, org_filter: str | None = None, verbose: bool = False
    ) -> dict:
        """
        Compatibility method: retrieve parents context and synthesize a response.

        Policy judging belongs in ``PolicyComplianceEvaluator``.
        """
        return self.query(
            question=text,
            collection=settings.qdrant.collection_parents,
            org_filter=org_filter,
            doc_type="policy",
            verbose=verbose,
        )

    def query_answer(
        self, question: str, org_filter: str | None = None, verbose: bool = False
    ) -> dict:
        """Compatibility method: retrieve children context and synthesize an answer."""
        return self.query(
            question=question,
            collection=settings.qdrant.collection_children,
            org_filter=org_filter,
            verbose=verbose,
        )
