"""Tests for query_engine.py — Conversion helpers and logging (no external calls)."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from query_engine import RAGQueryEngine


# ── Scored Points → Nodes ────────────────────────────────────────────────────

def _make_scored_point(text: str, score: float, **extra_payload):
    """Mimic a qdrant_client ScoredPoint."""
    payload = {"text": text, **extra_payload}
    return SimpleNamespace(payload=payload, score=score, id="fake-id")


class TestScoredPointsToNodes:
    def test_basic_conversion(self):
        points = [_make_scored_point("Hello world", 0.95, org="acme")]
        nodes = RAGQueryEngine._scored_points_to_nodes(points)
        assert len(nodes) == 1
        assert nodes[0].node.text == "Hello world"
        assert nodes[0].score == 0.95
        assert nodes[0].node.metadata["org"] == "acme"
        # text should NOT be in metadata
        assert "text" not in nodes[0].node.metadata

    def test_empty_input(self):
        nodes = RAGQueryEngine._scored_points_to_nodes([])
        assert nodes == []

    def test_multiple_points(self):
        points = [
            _make_scored_point("First", 0.9),
            _make_scored_point("Second", 0.8),
            _make_scored_point("Third", 0.7),
        ]
        nodes = RAGQueryEngine._scored_points_to_nodes(points)
        assert len(nodes) == 3
        texts = [n.node.text for n in nodes]
        assert texts == ["First", "Second", "Third"]

    def test_preserves_metadata(self):
        points = [_make_scored_point("text", 0.5, doc_id="D1", org="beta", source_file="a.pdf")]
        nodes = RAGQueryEngine._scored_points_to_nodes(points)
        meta = nodes[0].node.metadata
        assert meta["doc_id"] == "D1"
        assert meta["org"] == "beta"
        assert meta["source_file"] == "a.pdf"


# ── Query Log Writing ────────────────────────────────────────────────────────

class TestLogQuery:
    def test_writes_json_log(self, tmp_path):
        """Verify _log_query creates a valid JSON file in the logs directory."""

        # Patch the engine so it doesn't connect to Qdrant/Groq
        with patch.object(RAGQueryEngine, "__init__", lambda self: None):
            engine = RAGQueryEngine()
            engine.logs_dir = tmp_path

        engine._log_query(
            question="What is the policy?",
            collection="vocalmind_children",
            chunks=[{"rank": 1, "text": "Some policy text"}],
            response_text="The policy states ...",
            timing={"retrieval": 0.12, "synthesis": 0.34, "total": 0.46},
        )

        log_files = list(tmp_path.glob("query_*.json"))
        assert len(log_files) == 1

        with open(log_files[0], encoding="utf-8") as f:
            data = json.load(f)

        assert data["question"] == "What is the policy?"
        assert data["collection"] == "vocalmind_children"
        assert data["response"] == "The policy states ..."
        assert data["timing_seconds"]["total"] == 0.46
        assert len(data["retrieved_chunks"]) == 1
