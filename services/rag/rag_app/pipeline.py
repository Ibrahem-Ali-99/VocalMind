"""
Document ingestion pipeline for Generic RAG.

Handles loading, chunking, embedding, and indexing documents into a vector store.
Metadata extraction is done via loader-based heuristics (in ingestion strategies)
rather than LLM-based extraction, avoiding rate limits and latency.
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq
from pinecone import Pinecone, ServerlessSpec

from rag_app.config import settings
from rag_app.ingestion import get_ingestion_strategy


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents into the vector store.

    The pipeline uses a pluggable strategy pattern for document loading,
    allowing different data sources (HERB, RagBench, standard files) to be
    processed through the same chunking and indexing flow.
    """

    def __init__(self) -> None:
        """Initialize the ingestion pipeline with configured services."""
        settings.validate()
        self._setup_embedding_model()
        self._setup_llm()
        self._setup_pinecone()
        self.strategy = get_ingestion_strategy()

    def _setup_embedding_model(self) -> None:
        """Configure the Ollama embedding model."""
        self.embed_model = OllamaEmbedding(
            model_name=settings.embedding.model,
            base_url=settings.embedding.base_url,
        )

    def _setup_llm(self) -> None:
        """Configure Groq LLM (available for future use, e.g., query synthesis)."""
        self.llm = Groq(
            model=settings.groq.model,
            api_key=settings.groq.api_key.get_secret_value(),
            temperature=0.1,
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
                if "409" not in str(e) and "ALREADY_EXISTS" not in str(e):
                    raise e
                print(f"Index {settings.pinecone.index_name} already exists.")

        self.pinecone_index = self.pc.Index(settings.pinecone.index_name)

    def load_documents(self) -> list[Document]:
        """Load documents using the configured ingestion strategy."""
        return self.strategy.load_documents()

    def run(
        self, force_reindex: bool = False, documents: list[Document] | None = None
    ) -> VectorStoreIndex:
        """
        Execute the ingestion pipeline.

        Args:
            force_reindex: Currently unused, reserved for future cache invalidation.
            documents: Optional pre-loaded documents. If None, loads via strategy.

        Returns:
            A VectorStoreIndex connected to the Pinecone vector store.
        """
        # 1. Load Documents
        if documents is None:
            documents = self.load_documents()
        if not documents:
            print("No documents found to ingest.")
            return self._get_existing_index()

        print(f"Loaded {len(documents)} documents.")

        # 2. Build transformation pipeline (chunking + embedding)
        transformations = [
            SentenceSplitter(
                chunk_size=settings.chunking.chunk_size,
                chunk_overlap=settings.chunking.chunk_overlap,
            ),
            self.embed_model,
        ]

        # 3. Create Pipeline
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)

        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
        )

        # 4. Run Pipeline
        print("Starting ingestion pipeline...")
        nodes = pipeline.run(documents=documents, show_progress=True)
        print(f"Ingested {len(nodes)} nodes into Pinecone.")

        return VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self.embed_model
        )

    def _get_existing_index(self) -> VectorStoreIndex:
        """Connect to existing Pinecone index."""
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
        )
