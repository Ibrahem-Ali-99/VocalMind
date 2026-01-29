"""
Main entry point for the Generic RAG application.

Provides CLI interface for indexing, querying, and evaluation.
"""

import argparse
import sys
import os

from rag_app.pipeline import DocumentIngestionPipeline
from rag_app.query_engine import RAGQueryEngine
from rag_app.config import settings


def ingest_documents(force: bool = False, loader_type: str = None) -> None:
    """Run document ingestion pipeline."""
    if loader_type:
        os.environ["DATA_LOADER_TYPE"] = loader_type
        settings.reload()

    print("=" * 50)
    print(f"RAG Ingestion Pipeline (Loader: {settings.DATA_LOADER_TYPE})")
    print("=" * 50)

    pipeline = DocumentIngestionPipeline()
    pipeline.run(force_reindex=force)

    print("\n✓ Ingestion complete!")


def interactive_query() -> None:
    """Run interactive query session."""
    print("=" * 50)
    print("RAG Assistant")
    print("=" * 50)
    print("\nConnecting to index...")

    pipeline = DocumentIngestionPipeline()
    index = pipeline.run(force_reindex=False)

    engine = RAGQueryEngine(index)

    print("\n✓ Ready! Ask questions about your documents.")
    print("  Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            question = input("\n📝 Your question: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            print("\n🔍 Searching and generating response...\n")
            response = engine.query(question, verbose=True)
            print(f"\n💬 Answer:\n{response}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


def single_query(question: str) -> None:
    """Execute a single query and exit."""
    pipeline = DocumentIngestionPipeline()
    index = pipeline.run(force_reindex=False)

    engine = RAGQueryEngine(index)
    response = engine.query(question, verbose=True)

    print(f"\n{response}")


def run_evaluation(limit: int | None = None, product: str | None = None) -> None:
    """Run evaluation using Custom LLM Judge."""
    from rag_app.evaluation.custom_evaluator import CustomEvaluator
    from rag_app.pipeline import DocumentIngestionPipeline
    from llama_index.core import StorageContext, load_index_from_storage

    print("=" * 50)
    print("RAG Evaluation (Custom LLM Judge)")
    print("=" * 50)

    # Load index directly if possible, or via pipeline
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=str(settings.INDEX_DIR)
        )
        index = load_index_from_storage(storage_context)
    except Exception:
        print("Index not found, running ingestion first...")
        pipeline = DocumentIngestionPipeline()
        index = pipeline.run(force_reindex=False)

    evaluator = CustomEvaluator(index)
    questions = evaluator.load_test_data(limit=limit, product_filter=product)

    if not questions:
        print("No questions found to evaluate.")
        return

    report = evaluator.evaluate_batch(questions)
    evaluator.save_report(report)


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generic RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m rag_app.main                    # Interactive mode
  uv run python -m rag_app.main --ingest           # Index documents
  uv run python -m rag_app.main --ingest --force   # Force re-index
  uv run python -m rag_app.main -q "What is X?"    # Single query
  uv run python -m rag_app.main --evaluate         # Run RAGAS evaluation
  uv run python -m rag_app.main --evaluate --limit 5  # Run 5 questions
        """,
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion pipeline",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all documents",
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Execute a single query and exit",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAGAS evaluation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation questions",
    )

    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help="Filter evaluation to specific product",
    )

    args = parser.parse_args()

    try:
        if args.ingest:
            ingest_documents(force=args.force)
        elif args.query:
            single_query(args.query)
        if args.evaluate:
            run_evaluation(limit=args.limit, product=args.product)
        else:
            interactive_query()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
