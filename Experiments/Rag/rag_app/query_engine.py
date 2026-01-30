"""
Query engine module for Generic RAG.

Provides RAG query interface using Groq LLM and Pinecone retrieval.
"""

import json

import time
from datetime import datetime

from typing import Any
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq

from rag_app.config import settings


class RAGQueryEngine:
    """RAG query engine with logging."""

    def __init__(self, index: VectorStoreIndex) -> None:
        """Initialize query engine with vector index."""
        settings.validate()
        self.index = index
        self.logs_dir = settings.BASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self._setup_components()

    def _setup_components(self) -> None:
        """Configure LLM, retriever, and synthesizer."""
        self.llm = Groq(
            model=settings.groq.model,
            api_key=settings.groq.api_key.get_secret_value(),
            temperature=settings.groq.temperature,
            max_tokens=settings.groq.max_tokens,
            context_window=settings.groq.context_window,
        )

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.similarity_top_k,
        )

        self.synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=settings.response_mode,
        )

        # Generic system prompt
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=self.llm,
            context_prompt=(
                "You are a helpful assistant.\n"
                "Here are the relevant documents for the context:\n"
                "{context_str}\n"
                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            ),
            verbose=True,
            similarity_top_k=settings.similarity_top_k,
        )

    def _log_query(
        self, question: str, chunks: list[dict], response: str, timing: dict
    ) -> None:
        """Log query data to JSON file (runs in background thread)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"query_{timestamp}.json"

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "model": settings.groq.model,
            "similarity_top_k": settings.similarity_top_k,
            "timing_seconds": timing,
            "retrieved_chunks": chunks,
            "response": response,
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"\n📁 Query logged to: {log_file}")
        except Exception as e:
            print(f"Failed to log query: {e}")

    def _format_chunks(self, nodes: list) -> list[dict]:
        """Format retrieved nodes into serializable chunks."""
        chunks = []
        for i, node in enumerate(nodes, 1):
            metadata = node.metadata
            chunks.append(
                {
                    "rank": i,
                    "score": float(node.score) if node.score else None,
                    "metadata": metadata,
                    "text": node.text,
                    "text_length": len(node.text),
                }
            )
        return chunks

    def query(self, question: str, verbose: bool = False) -> Any:
        """Query the RAG system with timing and logging."""
        total_start = time.perf_counter()

        # Retrieval
        retrieval_start = time.perf_counter()
        nodes = self.retriever.retrieve(question)
        retrieval_time = time.perf_counter() - retrieval_start

        # Format chunks for display and logging
        chunks = self._format_chunks(nodes)

        # Display retrieved chunks
        if verbose:
            print("\n" + "=" * 60)
            print(f"📚 RETRIEVED CHUNKS ({retrieval_time:.2f}s)")
            print("=" * 60)

            for chunk in chunks:
                score_str = f"Score: {chunk['score']:.4f}" if chunk["score"] else ""
                print(f"\n[{chunk['rank']}] {score_str}")
                # Print metadata nicely
                meta_str = " | ".join(
                    f"{k}: {v}"
                    for k, v in chunk["metadata"].items()
                    if k != "file_path"
                )
                print(f"    Metadata: {meta_str}")
                print(
                    f"    Preview: {chunk['text'][: settings.chunk_preview_length]}..."
                )

            print("\n" + "=" * 60)

        # Synthesis
        synthesis_start = time.perf_counter()
        response = self.synthesizer.synthesize(question, nodes=nodes)
        synthesis_time = time.perf_counter() - synthesis_start

        total_time = time.perf_counter() - total_start
        if verbose:
            print(
                f"\n⏱️  Retrieval {retrieval_time:.2f}s | Synthesis {synthesis_time:.2f}s | Total {total_time:.2f}s"
            )

        response_text = str(response)

        # Inject timing into metadata for evaluation
        if response.metadata is None:
            response.metadata = {}
        response.metadata["retrieval_seconds"] = retrieval_time
        response.metadata["synthesis_seconds"] = synthesis_time
        response.metadata["total_seconds"] = total_time

        # Background logging
        timing_data = {
            "retrieval": round(retrieval_time, 4),
            "synthesis": round(synthesis_time, 4),
            "total": round(total_time, 4),
        }

        self._log_query(question, chunks, response_text, timing_data)

        return response
