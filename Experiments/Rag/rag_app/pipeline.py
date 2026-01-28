"""
Document ingestion pipeline for Generic RAG.

Handles loading (via strategy), chunking, and indexing documents into Pinecone.
"""

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag_app.config import settings
from rag_app.ingestion import get_ingestion_strategy


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into the vector store."""
    
    def __init__(self) -> None:
        """Initialize the ingestion pipeline."""
        settings.validate()
        self._setup_embedding_model()
        self._setup_pinecone()
        self.strategy = get_ingestion_strategy()
    
    def _setup_embedding_model(self) -> None:
        """Configure the Ollama embedding model."""
        self.embed_model = OllamaEmbedding(
            model_name=settings.embedding.model,
            base_url=settings.embedding.base_url,
        )
    
    def _setup_pinecone(self) -> None:
        """Initialize Pinecone client and create index if needed."""
        self.pc = Pinecone(api_key=settings.pinecone.api_key.get_secret_value())
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if settings.pinecone.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {settings.pinecone.index_name}")
            try:
                self.pc.create_index(
                    name=settings.pinecone.index_name,
                    dimension=settings.pinecone.dimension,
                    metric=settings.pinecone.metric,
                    spec=ServerlessSpec(
                        cloud=settings.pinecone.cloud,
                        region=settings.pinecone.region,
                    ),
                )
            except Exception as e:
                # 409 Conflict: Index already exists
                if "409" not in str(e) and "ALREADY_EXISTS" not in str(e):
                    raise e
                print(f"Index {settings.pinecone.index_name} already exists.")
        
        self.pinecone_index = self.pc.Index(settings.pinecone.index_name)
    
    def load_documents(self) -> list[Document]:
        """Load documents using the configured strategy."""
        return self.strategy.load_documents()
    
    def create_index(self, documents: list[Document]) -> VectorStoreIndex:
        """Create vector index from documents."""
        if not documents:
            print("No documents to index.")
            # Return empty index conencted to store
            self.get_existing_index()
            
        splitter = SentenceSplitter(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print(f"Indexing {len(documents)} documents into Pinecone...")
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            transformations=[splitter],
            show_progress=True,
        )
        
        print("Indexing complete!")
        return index
    
    def get_existing_index(self) -> VectorStoreIndex:
        """Connect to existing Pinecone index."""
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
        )
    
    def run(self, force_reindex: bool = False) -> VectorStoreIndex:
        """Execute ingestion pipeline."""
        stats = self.pinecone_index.describe_index_stats()
        
        if stats.total_vector_count > 0 and not force_reindex:
            print(f"Using existing index with {stats.total_vector_count} vectors")
            return self.get_existing_index()
        
        documents = self.load_documents()
        if not documents:
             print("No documents found to ingest.")
             return self.get_existing_index()
             
        return self.create_index(documents)
