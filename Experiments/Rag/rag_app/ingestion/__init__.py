"""
Ingestion strategies package.
"""

from rag_app.ingestion.base import BaseIngestionStrategy
from rag_app.ingestion.standard import StandardIngestionStrategy
from rag_app.ingestion.herb import HERBIngestionStrategy
from rag_app.config import settings

def get_ingestion_strategy() -> BaseIngestionStrategy:
    """Factory to get the configured ingestion strategy."""
    loader_type = settings.DATA_LOADER_TYPE.lower()
    
    if loader_type == "herb":
        return HERBIngestionStrategy()
    elif loader_type == "standard":
        return StandardIngestionStrategy()
    else:
        print(f"Unknown loader type '{loader_type}', defaulting to standard.")
        return StandardIngestionStrategy()
