"""Tests for config.py — Settings defaults and sub-config validation."""

import os
from pathlib import Path

import pytest

# Ensure test env vars are set before importing config
os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from config import (
    ChildChunkingConfig,
    EmbeddingConfig,
    GroqConfig,
    ParentChunkingConfig,
    QdrantConfig,
    Settings,
)


class TestGroqConfig:
    def test_default_model(self):
        cfg = GroqConfig(GROQ_API_KEY="test")
        assert cfg.model == "llama-3.3-70b-versatile"

    def test_default_temperature(self):
        cfg = GroqConfig(GROQ_API_KEY="test")
        assert cfg.temperature == 0.1

    def test_api_key_is_secret(self):
        cfg = GroqConfig(GROQ_API_KEY="my_secret")
        assert cfg.api_key.get_secret_value() == "my_secret"
        # Should not leak in string representation
        assert "my_secret" not in str(cfg.api_key)


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == "snowflake-arctic-embed2"
        assert cfg.dimension == 1024
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.request_timeout == 120.0


class TestQdrantConfig:
    def test_defaults(self):
        cfg = QdrantConfig()
        assert cfg.url == "http://localhost:6333"
        assert cfg.collection_parents == "vocalmind_parents"
        assert cfg.collection_children == "vocalmind_children"


class TestParentChunkingConfig:
    def test_headers(self):
        cfg = ParentChunkingConfig()
        assert len(cfg.headers_to_split_on) == 3
        markers = [h[0] for h in cfg.headers_to_split_on]
        assert markers == ["#", "##", "###"]

    def test_empty_section_min_words(self):
        cfg = ParentChunkingConfig()
        assert cfg.empty_section_min_words == 4


class TestChildChunkingConfig:
    def test_defaults(self):
        cfg = ChildChunkingConfig()
        assert cfg.chunk_size == 400
        assert cfg.chunk_overlap == 80
        assert cfg.min_chunk_length == 30


class TestSettings:
    def test_base_dir_is_path(self):
        s = Settings()
        assert isinstance(s.BASE_DIR, Path)

    def test_response_mode(self):
        s = Settings()
        assert s.response_mode in ("compact", "refine", "tree_summarize")

    def test_similarity_top_k(self):
        s = Settings()
        assert s.similarity_top_k == 5

    def test_validate_config_raises_on_missing_docs_dir(self, tmp_path):
        s = Settings()
        s.DOCS_DIR = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="does not exist"):
            s.validate_config()

    def test_validate_config_passes_with_existing_dir(self, tmp_path):
        s = Settings()
        s.DOCS_DIR = tmp_path
        s.validate_config()  # Should not raise
