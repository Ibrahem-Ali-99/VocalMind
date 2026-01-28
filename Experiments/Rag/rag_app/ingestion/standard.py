"""
Standard file ingestion strategy.
"""

from typing import List

from llama_index.core import Document, SimpleDirectoryReader
from rag_app.config import settings
from rag_app.ingestion.base import BaseIngestionStrategy


class StandardIngestionStrategy(BaseIngestionStrategy):
    """Ingests standard files (PDF, TXT, HTML, etc.) from a directory."""
    
    def load_documents(self) -> List[Document]:
        data_dir = settings.DATA_DIR
        
        if not data_dir.exists():
            # If default data dir doesn't exist, try to create it or warn
            print(f"Warning: Data directory {data_dir} does not exist.")
            return []
            
        print(f"Loading standard documents from: {data_dir}")
        
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            recursive=True,
            # Excluding hidden files and specific metadata files if in mixed dir
            exclude_hidden=True,
        )
        
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents.")
        return documents
