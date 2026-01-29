import os
from rag_app.config import settings
from rag_app.pipeline import DocumentIngestionPipeline
from rag_app.evaluation.ragas_evaluator import RAGASEvaluator


def main():
    print("=" * 50)
    print("Running RAGAS Evaluation (Subset: ActionGenie)")
    print("=" * 50)

    # 1. Setup Environment
    # We will manually load data, so DATA_LOADER_TYPE doesn't matter much for loading,
    # but we set RAG_DATA_DIR for questions loading.
    os.environ["DATA_LOADER_TYPE"] = "herb"
    os.environ["RAG_DATA_DIR"] = "data/HERB"
    # Metadata extraction is now HEURISTIC based (efficient), so we don't need to disable it.
    settings.reload()

    # 2. Skip Ingestion (Data is already in Pinecone from previous run)
    print("\n[1/3] Connecting to existing Index (Skipping Ingestion)...")
    pipeline = DocumentIngestionPipeline()
    # just get the index object
    index = pipeline._get_existing_index()

    # 3. Setup Evaluator
    print("\n[2/3] Preparing Evaluator...")
    evaluator = RAGASEvaluator(index)

    # 4. Load Test Questions (Fetch 10, take next 5)
    # We tested [0:3] previously. Now we want [3:8].
    all_questions = evaluator.load_herb_data(limit=10, product_filter="ActionGenie")
    if len(all_questions) < 8:
        print(f"Warning: Only found {len(all_questions)} questions total.")

    questions = all_questions[
        3:8
    ]  # standard python slicing is safe even if out of bounds

    if not questions:
        print("❌ No questions found for ActionGenie.")
        return

    # 5. Run Evaluation
    print(f"\n[3/3] Evaluating {len(questions)} NEW questions with RAGAS...")
    report = evaluator.evaluate_batch(questions)

    # 6. Save Report
    evaluator.save_report(report)
    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
