"""
Configuration module using Pydantic Settings.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

load_dotenv()


class GroqConfig(BaseSettings):
    api_key: SecretStr = Field(alias="GROQ_API_KEY")
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 131072


class EmbeddingConfig(BaseSettings):
    model: str = "embeddinggemma:latest"
    base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    embed_batch_size: int = 10


class PineconeConfig(BaseSettings):
    api_key: SecretStr = Field(alias="PINECONE_API_KEY")
    index_name: str = Field(default="rag-app-index", alias="PINECONE_INDEX_NAME")
    dimension: int = 768
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class ChunkingConfig(BaseSettings):
    chunk_size: int = 1024
    chunk_overlap: int = 200


class Settings(BaseSettings):
    """Application-wide settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__"
    )

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        alias="RAG_DATA_DIR"
    )

    # Core Settings
    DATA_LOADER_TYPE: Literal["standard", "herb"] = "standard"
    
    # Sub-configs
    groq: GroqConfig = Field(default_factory=GroqConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    
    # App Settings
    similarity_top_k: int = 3
    response_mode: str = "compact"
    chunk_preview_length: int = 150

    def reload(self) -> None:
        """Reload settings from environment variables."""
        # Create a new instance with fresh env vars
        new_settings = Settings()
        # Update current instance attributes
        self.__dict__.update(new_settings.__dict__)

    def validate(self) -> None:
        """Pydantic validation happens automatically on init."""
        pass


settings = Settings()
