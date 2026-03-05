"""
Configuration module for VocalMind Final RAG.

Uses Pydantic Settings for type-safe config with .env file support.
Architecture:
  Parsing    → Docling  (AI-powered PDF → Markdown)
  Embeddings → Ollama   (snowflake-arctic-embed2, 1024-dim)
  Vector DB  → Qdrant   (dual collections: parents + children)
  LLM        → Groq     (fast cloud inference)
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


# ── Sub-configs ───────────────────────────────────────────────────────────────


class GroqConfig(BaseSettings):
    """Groq LLM configuration."""

    api_key: SecretStr = Field(alias="GROQ_API_KEY")
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 131_072


class EmbeddingConfig(BaseSettings):
    """Ollama embedding model configuration."""

    model: str = Field(default="snowflake-arctic-embed2", alias="EMBEDDING_MODEL")
    base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    dimension: int = 1024  # snowflake-arctic-embed2 output dimension
    request_timeout: float = 120.0


class QdrantConfig(BaseSettings):
    """Qdrant vector store configuration."""

    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    collection_parents: str = "vocalmind_parents"
    collection_children: str = "vocalmind_children"


class ParentChunkingConfig(BaseSettings):
    """Parent chunking: Markdown header splitting on H1/H2/H3."""

    headers_to_split_on: list[tuple[str, str]] = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    empty_section_min_words: int = 4


class ChildChunkingConfig(BaseSettings):
    """Child chunking: Recursive character splitting for precision snippets."""

    chunk_size: int = 400
    chunk_overlap: int = 80
    min_chunk_length: int = 30


# ── Main Settings ─────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application-wide settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DOCS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent / "docs",
        alias="DOCS_DIR",
    )
    PARSED_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent / "parsed_docs",
        alias="PARSED_DIR",
    )

    # Sub-configs
    groq: GroqConfig = Field(default_factory=GroqConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    parent_chunking: ParentChunkingConfig = Field(default_factory=ParentChunkingConfig)
    child_chunking: ChildChunkingConfig = Field(default_factory=ChildChunkingConfig)

    # Query defaults
    similarity_top_k: int = 5
    response_mode: Literal["compact", "refine", "tree_summarize"] = "compact"

    def validate_config(self) -> None:
        """Validate critical config at startup."""
        if not self.DOCS_DIR.exists():
            raise ValueError(f"Docs directory {self.DOCS_DIR} does not exist.")


settings = Settings()
