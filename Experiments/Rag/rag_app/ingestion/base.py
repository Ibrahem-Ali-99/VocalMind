"""
Base ingestion strategy interface.
"""

from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Document


class BaseIngestionStrategy(ABC):
    """Abstract base class for data ingestion strategies."""
    
    @abstractmethod
    def load_documents(self) -> List[Document]:
        """Load documents from the configured data source."""
        pass
