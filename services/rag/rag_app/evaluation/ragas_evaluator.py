"""
RAGAS-based evaluation for the RAG pipeline.

Provides faithfulness, answer_relevancy, and context_recall metrics.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex

from rag_app.config import settings
from rag_app.evaluation.models import EvaluationSample, EvaluationReport, TimingMetrics
from rag_app.query_engine import RAGQueryEngine


class RAGASEvaluator:
    """Evaluator that runs RAGAS metrics on the RAG pipeline."""

    def __init__(self, index: VectorStoreIndex) -> None:
        """Initialize evaluator with a vector index."""
        self.index = index
        self.query_engine = RAGQueryEngine(index)
        self.reports_dir = settings.BASE_DIR / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def _query_with_timing(self, question: str) -> tuple[str, list[str], TimingMetrics]:
        """Execute query and capture timing metrics."""
        total_start = time.perf_counter()

        # Retrieval
        retrieval_start = time.perf_counter()
        nodes = self.query_engine.retriever.retrieve(question)
        retrieval_time = time.perf_counter() - retrieval_start

        # Extract contexts
        contexts = [node.text for node in nodes]

        # Synthesis
        synthesis_start = time.perf_counter()
        response = self.query_engine.synthesizer.synthesize(question, nodes=nodes)
        synthesis_time = time.perf_counter() - synthesis_start

        total_time = time.perf_counter() - total_start

        timing = TimingMetrics(
            retrieval_seconds=round(retrieval_time, 4),
            synthesis_seconds=round(synthesis_time, 4),
            total_seconds=round(total_time, 4),
        )

        return str(response), contexts, timing

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str | list[str] | None = None,
    ) -> EvaluationSample:
        """Evaluate a single question and return metrics."""
        answer, contexts, timing = self._query_with_timing(question)

        sample = EvaluationSample(
            question=question,
            generated_answer=answer,
            ground_truth=ground_truth,
            retrieved_contexts=contexts,
            timing=timing,
        )

        # Compute RAGAS metrics if ragas is available
        try:
            sample = self._compute_ragas_metrics(sample)
        except ImportError:
            print("RAGAS not installed, skipping metric computation.")
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")

        return sample

    def _compute_ragas_metrics(self, sample: EvaluationSample) -> EvaluationSample:
        """Compute RAGAS metrics for a sample."""
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas import EvaluationDataset, SingleTurnSample
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Initialize LLM
        groq_llm = ChatGroq(
            model=settings.groq.model,
            api_key=settings.groq.api_key.get_secret_value(),
            temperature=0.0,
        )
        evaluator_llm = LangchainLLMWrapper(langchain_llm=groq_llm)

        # Initialize Embeddings
        ollama_embeddings = OllamaEmbeddings(
            base_url=settings.embedding.base_url,
            model=settings.embedding.model,
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=ollama_embeddings)

        # Prepare ground truth
        reference = sample.ground_truth
        if isinstance(reference, list):
            reference = ", ".join(reference)
        if reference is None:
            reference = ""

        # Create RAGAS sample
        ragas_sample = SingleTurnSample(
            user_input=sample.question,
            response=sample.generated_answer,
            retrieved_contexts=sample.retrieved_contexts,
            reference=reference,
        )

        dataset = EvaluationDataset(samples=[ragas_sample])

        # Select metrics based on available data
        metrics = [faithfulness, answer_relevancy]
        if reference:
            metrics.append(context_recall)

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        scores = result.to_pandas().iloc[0].to_dict()

        sample.faithfulness = scores.get("faithfulness")
        sample.answer_relevancy = scores.get("answer_relevancy")
        sample.context_recall = scores.get("context_recall")

        return sample

    def load_herb_data(
        self,
        limit: int | None = None,
        product_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load questions from HERB dataset."""
        herb_dir = settings.DATA_DIR
        if not (herb_dir / "products").exists():
            herb_dir = settings.BASE_DIR / "HERB"

        products_dir = herb_dir / "products"
        if not products_dir.exists():
            print(f"Warning: HERB products not found at {products_dir}")
            return []

        questions: list[dict[str, Any]] = []

        for product_file in products_dir.glob("*.json"):
            if (
                product_filter
                and product_filter.lower() not in product_file.stem.lower()
            ):
                continue

            try:
                with open(product_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # Load answerable questions
            for q in data.get("answerable_questions", []):
                questions.append(
                    {
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                        "product": product_file.stem,
                    }
                )

        if limit:
            questions = questions[:limit]

        print(f"Loaded {len(questions)} questions from HERB dataset")
        return questions

    def evaluate_batch(
        self,
        questions: list[dict[str, Any]],
    ) -> EvaluationReport:
        """Evaluate a batch of questions."""
        samples: list[EvaluationSample] = []

        for i, q in enumerate(questions, 1):
            print(
                f"[{i}/{len(questions)}] Converting {q['product']}: {q['question'][:50]}..."
            )
            sample = self.evaluate_sample(
                question=q["question"],
                ground_truth=q.get("ground_truth"),
            )
            samples.append(sample)

        return self._create_report(samples)

    def _create_report(self, samples: list[EvaluationSample]) -> EvaluationReport:
        """Aggregate samples into a report."""
        if not samples:
            raise ValueError("No samples to report")

        # Timing averages
        avg_retrieval = sum(s.timing.retrieval_seconds for s in samples) / len(samples)
        avg_synthesis = sum(s.timing.synthesis_seconds for s in samples) / len(samples)
        avg_total = sum(s.timing.total_seconds for s in samples) / len(samples)

        # RAGAS metric averages (only for non-None values)
        faith_scores = [s.faithfulness for s in samples if s.faithfulness is not None]
        relevancy_scores = [
            s.answer_relevancy for s in samples if s.answer_relevancy is not None
        ]
        recall_scores = [
            s.context_recall for s in samples if s.context_recall is not None
        ]

        avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None
        avg_rel = (
            sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
        )
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None

        print("\nEvaluation Results:")
        print(
            f"  Faithfulness:      {avg_faith:.4f}"
            if avg_faith
            else "  Faithfulness:      N/A"
        )
        print(
            f"  Answer Relevancy:  {avg_rel:.4f}"
            if avg_rel
            else "  Answer Relevancy:  N/A"
        )
        print(
            f"  Context Recall:    {avg_recall:.4f}"
            if avg_recall
            else "  Context Recall:    N/A"
        )

        report = EvaluationReport(
            model=settings.groq.model,
            total_samples=len(samples),
            avg_retrieval_seconds=round(avg_retrieval, 4),
            avg_synthesis_seconds=round(avg_synthesis, 4),
            avg_total_seconds=round(avg_total, 4),
            avg_faithfulness=round(avg_faith, 4) if avg_faith is not None else None,
            avg_answer_relevancy=round(avg_rel, 4) if avg_rel is not None else None,
            avg_context_recall=round(avg_recall, 4) if avg_recall is not None else None,
            samples=samples,
        )

        return report

    def save_report(
        self, report: EvaluationReport, filename: str | None = None
    ) -> Path:
        """Save report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_report_{timestamp}.json"

        filepath = self.reports_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)

        print(f"Report saved to: {filepath}")
        return filepath
