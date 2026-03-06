"""Tests for ingest.py — Pure logic: cleaning, chunking, metadata, validation."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from ingest import DocumentIngestionPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_chunk(content: str, metadata: dict | None = None):
    """Create a SimpleNamespace mimicking a langchain Document chunk."""
    return SimpleNamespace(page_content=content, metadata=metadata or {})


# Use a module-level fixture to avoid connecting to Qdrant/Ollama on init
@pytest.fixture()
def pipeline():
    """Return a pipeline object without connecting to external services."""
    with patch.object(DocumentIngestionPipeline, "__init__", lambda self: None):
        p = DocumentIngestionPipeline()
        return p


# ── Text Cleaning ─────────────────────────────────────────────────────────────

class TestFixEncoding:
    def test_html_entities(self, pipeline):
        assert pipeline._fix_encoding("&amp; &lt; &gt;") == "& < >"

    def test_html_hex_entities(self, pipeline):
        assert pipeline._fix_encoding("&#x26; &#x3C; &#x3E;") == "& < >"

    def test_quote_entities(self, pipeline):
        assert pipeline._fix_encoding("&quot; &#x27;") == '" \''

    def test_passthrough(self, pipeline):
        text = "Normal text without entities."
        assert pipeline._fix_encoding(text) == text


class TestRepairOrphanedTableRows:
    def test_repairs_separated_rows(self, pipeline):
        text = "| A | B |\n\n| 1 | 2 |"
        result = pipeline._repair_orphaned_table_rows(text)
        assert "| A | B |\n| 1 | 2 |" in result

    def test_no_change_for_non_table(self, pipeline):
        text = "Hello\n\nWorld"
        assert pipeline._repair_orphaned_table_rows(text) == text


class TestCleanMarkdown:
    def test_combined(self, pipeline):
        text = "| A &amp; B |\n\n| 1 | 2 |"
        result = pipeline._clean_markdown(text)
        assert "&amp;" not in result
        assert "& B" in result


# ── Metadata Extraction ──────────────────────────────────────────────────────

class TestExtractMetadata:
    def test_extracts_fields(self):
        md = (
            "**Organization**: Acme Corp\n"
            "**Department**: Support\n"
            "**Document ID**: DOC-001\n"
            "**Version**: 2.0\n"
            "**Effective Date**: 2025-01-01\n"
        )
        meta = DocumentIngestionPipeline._extract_metadata(md, "test.pdf", "acme")
        assert meta["org"] == "acme"
        assert meta["department"] == "Support"
        assert meta["doc_id"] == "DOC-001"
        assert meta["version"] == "2.0"
        assert meta["effective_date"] == "2025-01-01"
        assert meta["source_file"] == "test.pdf"
        assert "ingested_at" in meta

    def test_defaults_to_unknown(self):
        meta = DocumentIngestionPipeline._extract_metadata("No metadata here.", "f.pdf", "")
        assert meta["department"] == "Unknown"
        assert meta["doc_id"] == "Unknown"

    def test_org_override_from_folder(self):
        md = "**Organization**: In-Doc Org\n"
        meta = DocumentIngestionPipeline._extract_metadata(md, "f.pdf", "folder_org")
        assert meta["org"] == "folder_org"


# ── Table Detection ──────────────────────────────────────────────────────────

class TestIsTableChunk:
    def test_pipe_table(self):
        text = "| Col A | Col B |\n|---|---|\n| 1 | 2 |"
        assert DocumentIngestionPipeline._is_table_chunk(text) is True

    def test_html_table(self):
        text = "<table><tr><td>A</td></tr></table>"
        assert DocumentIngestionPipeline._is_table_chunk(text) is True

    def test_non_table(self):
        text = "Just some paragraph text."
        assert DocumentIngestionPipeline._is_table_chunk(text) is False


# ── Empty Section Detection ──────────────────────────────────────────────────

class TestIsEmptySection:
    def test_empty(self):
        chunk = _make_fake_chunk("## Title")
        assert DocumentIngestionPipeline._is_empty_section(chunk) is True

    def test_not_empty(self):
        chunk = _make_fake_chunk("## Title\nThis section has enough content to pass the check.")
        assert DocumentIngestionPipeline._is_empty_section(chunk) is False


# ── Chunk Validation ─────────────────────────────────────────────────────────

class TestValidateChunks:
    def test_no_warnings(self):
        chunks = [_make_fake_chunk("A" * 50), _make_fake_chunk("B" * 50)]
        report = DocumentIngestionPipeline._validate_chunks(chunks, "TEST")
        assert report["total"] == 2
        assert report["short"] == 0
        assert report["duplicates"] == 0
        assert report["warnings"] == []

    def test_short_chunk_warning(self):
        chunks = [_make_fake_chunk("Hi")]
        report = DocumentIngestionPipeline._validate_chunks(chunks, "TEST")
        assert report["short"] == 1
        assert len(report["warnings"]) == 1

    def test_duplicate_warning(self):
        chunks = [_make_fake_chunk("A" * 50), _make_fake_chunk("A" * 50)]
        report = DocumentIngestionPipeline._validate_chunks(chunks, "TEST")
        assert report["duplicates"] == 1
        assert len(report["warnings"]) == 1
