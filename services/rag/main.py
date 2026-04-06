"""
VocalMind Final RAG — CLI Entry Point.

Subcommands:
  --ingest           Ingest PDFs from docs/ into Qdrant (parents + children)
  --ingest --force   Wipe collections and re-ingest
    --watch            Keep running and reprocess documents on change
  -q QUESTION        Single query (defaults to children collection)
  --compliance TEXT   Check policy compliance of a transcript
  --check-answer     Check correctness of an agent's answer
  (default)          Interactive mode

Examples:
  python main.py --ingest
  python main.py --ingest --force
    python main.py --watch
  python main.py -q "What is the refund policy?"
  python main.py -q "What is the refund policy?" --org org1
  python main.py --compliance "Agent said: sure, full refund, no questions"
  python main.py --check-answer --question "Refund window?" --answer "30 days"
  python main.py                          # interactive mode
"""

import argparse
import sys
import time
from pathlib import Path


def _snapshot_documents(docs_dirs: list[Path]) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for docs_dir in docs_dirs:
        if not docs_dir.exists():
            continue

        for pattern in ("*.pdf", "*.PDF"):
            for pdf in docs_dir.rglob(pattern):
                if not pdf.is_file():
                    continue
                stat = pdf.stat()
                snapshot[str(pdf.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run document ingestion pipeline."""
    try:
        from .ingest import DocumentIngestionPipeline
    except ImportError:  # pragma: no cover - direct script execution
        from ingest import DocumentIngestionPipeline

    print("=" * 60)
    print("VocalMind Final RAG — Document Ingestion")
    print("=" * 60)

    pipeline = DocumentIngestionPipeline()
    pipeline.run(force=args.force)

    print("\nIngestion complete!")


def cmd_watch(args: argparse.Namespace) -> None:
    """Keep the ingestion service alive and reprocess documents on change."""
    try:
        from .config import settings
        from .ingest import DocumentIngestionPipeline
    except ImportError:  # pragma: no cover - direct script execution
        from config import settings
        from ingest import DocumentIngestionPipeline

    print("=" * 60)
    print("VocalMind Final RAG — Watch Mode")
    print("=" * 60)

    pipeline = DocumentIngestionPipeline()
    candidate_dirs = [
        Path(settings.DOCS_DIR),
        Path(settings.POLICY_DOCS_DIR),
        Path(settings.KNOWLEDGE_DOCS_DIR),
    ]
    docs_dirs: list[Path] = []
    for path in candidate_dirs:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in docs_dirs:
            continue
        docs_dirs.append(resolved)

    interval = max(10, args.interval)
    last_snapshot: dict[str, tuple[int, int]] = {}

    while True:
        current_snapshot = _snapshot_documents(docs_dirs)
        if current_snapshot != last_snapshot:
            print("\nDocument changes detected. Running ingestion...\n")
            pipeline.run(force=args.force)
            last_snapshot = current_snapshot
        else:
            print(f"\nNo document changes detected. Sleeping for {interval}s...")

        time.sleep(interval)


def cmd_query(args: argparse.Namespace) -> None:
    """Execute a single query."""
    try:
        from .config import settings
        from .query_engine import RAGQueryEngine
    except ImportError:  # pragma: no cover - direct script execution
        from config import settings
        from query_engine import RAGQueryEngine

    engine = RAGQueryEngine()

    collection = (
        settings.qdrant.collection_parents
        if args.parents
        else settings.qdrant.collection_children
    )

    result = engine.query(
        question=args.query,
        collection=collection,
        org_filter=args.org,
        verbose=True,
    )

    print(f"\nAnswer:\n{result['response']}")


def cmd_compliance(args: argparse.Namespace) -> None:
    """Run a policy compliance check."""
    try:
        from .evaluator import PolicyComplianceEvaluator
    except ImportError:  # pragma: no cover - direct script execution
        from evaluator import PolicyComplianceEvaluator

    print("=" * 60)
    print("VocalMind Final RAG — Policy Compliance Check")
    print("=" * 60)

    evaluator = PolicyComplianceEvaluator()
    result = evaluator.check(
        transcript=args.compliance,
        org_filter=args.org,
        verbose=True,
    )

    status = "PASS" if result.compliance_score >= 0.7 else "FAIL"
    print(f"\n{'='*60}")
    print(f"Compliance Score: {result.compliance_score:.2f}  [{status}]")
    print(f"Reasoning: {result.reasoning}")
    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"  - {v}")
    if result.policy_references:
        print("\nPolicy References:")
        for p in result.policy_references:
            print(f"  - {p}")


def cmd_check_answer(args: argparse.Namespace) -> None:
    """Run an answer correctness check."""
    try:
        from .evaluator import AnswerCorrectnessEvaluator
    except ImportError:  # pragma: no cover - direct script execution
        from evaluator import AnswerCorrectnessEvaluator

    print("=" * 60)
    print("VocalMind Final RAG — Answer Correctness Check")
    print("=" * 60)

    evaluator = AnswerCorrectnessEvaluator()
    result = evaluator.check(
        question=args.question,
        agent_answer=args.answer,
        org_filter=args.org,
        verbose=True,
    )

    icon = "CORRECT" if result.is_correct else "INCORRECT"
    print(f"\n{'='*60}")
    print(f"Correctness Score: {result.correctness_score:.2f}  [{icon}]")
    print(f"Reasoning: {result.reasoning}")
    if result.source_references:
        print("\nSources:")
        for s in result.source_references:
            print(f"  - {s}")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive query session."""
    try:
        from .config import settings
        from .query_engine import RAGQueryEngine
    except ImportError:  # pragma: no cover - direct script execution
        from config import settings
        from query_engine import RAGQueryEngine

    print("=" * 60)
    print("VocalMind Final RAG — Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  /parents    — switch to parents collection (compliance)")
    print("  /children   — switch to children collection (answers)")
    print("  /org NAME   — filter by organization")
    print("  /org        — clear org filter")
    print("  /quit       — exit")
    print()

    engine = RAGQueryEngine()
    collection = settings.qdrant.collection_children
    org_filter = args.org if hasattr(args, "org") else None

    while True:
        try:
            prompt_label = f"[{collection.split('_')[-1]}]"
            if org_filter:
                prompt_label += f" [{org_filter}]"
            user_input = input(f"\n{prompt_label} Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "quit", "exit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "/parents":
                collection = settings.qdrant.collection_parents
                print(f"  Switched to: {collection}")
                continue

            if user_input.lower() == "/children":
                collection = settings.qdrant.collection_children
                print(f"  Switched to: {collection}")
                continue

            if user_input.lower().startswith("/org"):
                parts = user_input.split(maxsplit=1)
                org_filter = parts[1].strip() if len(parts) > 1 else None
                print(f"  Org filter: {org_filter or '(cleared)'}")
                continue

            result = engine.query(
                question=user_input,
                collection=collection,
                org_filter=org_filter,
                verbose=True,
            )
            print(f"\nAnswer:\n{result['response']}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VocalMind Final RAG — Policy Compliance & Answer Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--ingest", action="store_true",
        help="Run document ingestion pipeline",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-indexing (wipe collections and re-ingest)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Keep the ingestion pipeline running and reprocess on document changes",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Polling interval in seconds for --watch",
    )
    parser.add_argument(
        "-q", "--query", type=str,
        help="Execute a single query and exit",
    )
    parser.add_argument(
        "--parents", action="store_true",
        help="Query parents collection instead of children (for -q flag)",
    )
    parser.add_argument(
        "--compliance", type=str,
        help="Check policy compliance of a transcript",
    )
    parser.add_argument(
        "--check-answer", action="store_true",
        help="Check correctness of an agent's answer",
    )
    parser.add_argument(
        "--question", type=str,
        help="Customer question (used with --check-answer)",
    )
    parser.add_argument(
        "--answer", type=str,
        help="Agent's answer (used with --check-answer)",
    )
    parser.add_argument(
        "--org", type=str, default=None,
        help="Filter retrieval to a specific organization",
    )

    args = parser.parse_args()

    try:
        if args.ingest:
            cmd_ingest(args)
        elif args.watch:
            cmd_watch(args)
        elif args.query:
            cmd_query(args)
        elif args.compliance:
            cmd_compliance(args)
        elif args.check_answer:
            if not args.question or not args.answer:
                parser.error("--check-answer requires --question and --answer")
            cmd_check_answer(args)
        else:
            cmd_interactive(args)

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\nConnection Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
