from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Document


class BaseIngestionStrategy(ABC):
    """Abstract base class for ingestion strategies."""

    @abstractmethod
    def load_documents(self) -> List[Document]:
        """Load documents from source."""
        pass
