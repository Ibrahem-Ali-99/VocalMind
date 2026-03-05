from rag_app.config import settings
from rag_app.ingestion.base import BaseIngestionStrategy as BaseIngestionStrategy
from rag_app.ingestion.herb import HERBIngestionStrategy
from rag_app.ingestion.standard import StandardIngestionStrategy

# Import new loaders
from rag_app.ingestion.loaders import CustomDirectoryLoader, RagBenchLoader


def get_ingestion_strategy():
    """Factory to get the configured ingestion strategy/loader."""
    loader_type = settings.DATA_LOADER_TYPE

    if loader_type == "herb":
        return HERBIngestionStrategy()
    elif loader_type == "ragbench":
        return RagBenchLoader()
    elif loader_type == "directory":
        # Return a simple wrapper that matches the expected interface (load_documents)
        return DirectoryStrategyWrapper()
    else:
        return StandardIngestionStrategy()


class DirectoryStrategyWrapper:
    """Wrapper for CustomDirectoryLoader to match BaseIngestionStrategy interface."""

    def load_documents(self):
        loader = CustomDirectoryLoader(
            input_dir=str(settings.DATA_DIR),
            recursive=True,
            required_exts=[".pdf", ".txt", ".md", ".json", ".docx"],
        )
        return loader.load_data()
