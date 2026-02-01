"""
Custom evaluator for RAG system using direct LLM-as-a-Judge.
"""

import json
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rag_app.config import settings
from rag_app.evaluation.models import EvaluationSample, EvaluationReport, TimingMetrics
from rag_app.query_engine import RAGQueryEngine
from llama_index.core import VectorStoreIndex


class CustomEvaluator:
    def __init__(self, index: VectorStoreIndex):
        self.query_engine = RAGQueryEngine(index)
        self.llm = ChatGroq(
            model=settings.groq.model,
            api_key=settings.groq.api_key.get_secret_value(),
            temperature=0.0,
        )

    def load_test_data(
        self,
        limit: int | None = None,
        product_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load questions, filtering out ID-based queries to ensure fair evaluation."""
        herb_dir = settings.DATA_DIR
        if not (herb_dir / "products").exists():
            herb_dir = settings.BASE_DIR / "HERB"

        products_dir = herb_dir / "products"
        questions: list[dict[str, Any]] = []

        if not products_dir.exists():
            print(f"Warning: HERB products not found at {products_dir}")
            return []

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

            for q in data.get("answerable_questions", []):
                # Filter out Employee ID / Utterance ID / PR ID questions
                q_text = q["question"].lower()
                if (
                    "employee id" in q_text
                    or "utterance id" in q_text
                    or "pr id" in q_text
                ):
                    continue

                questions.append(
                    {
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                        "product": product_file.stem,
                    }
                )

        if limit:
            questions = questions[:limit]

        print(f"Loaded {len(questions)} evaluatable questions (filtered IDs)")
        return questions

    def evaluate_answer(
        self, question: str, generated_answer: str, ground_truth: Any
    ) -> tuple[bool, str]:
        """Use LLM to judge if the generated answer matches ground truth."""
        try:
            prompt = ChatPromptTemplate.from_template("""
                You are an impartial judge evaluating a RAG system.
                
                Question: {question}
                Generated Answer: {generated_answer}
                Ground Truth: {ground_truth}
                
                Does the generated answer contain the core information present in the Ground Truth?
                If the Ground Truth is a list of items, the answer should mention the relevant ones.
                If the Ground Truth is 'Not Found' and the answer says 'I don't know', that is CORRECT.
                
                Respond with JSON only:
                {{
                    "is_correct": boolean,
                    "reason": "short explanation"
                }}
            """)

            chain = prompt | self.llm
            response = chain.invoke(
                {
                    "question": question,
                    "generated_answer": generated_answer,
                    "ground_truth": str(ground_truth),
                }
            )

            content = response.content.strip()
            # Handle markdown code blocks if present
            if "```" in content:
                # Try to extract JSON from code block
                parts = content.split("```")
                # Look for the part that looks like JSON
                for part in parts:
                    clean_part = part.replace("json", "").strip()
                    if clean_part.startswith("{"):
                        content = clean_part
                        break

            result = json.loads(content)
            return result.get("is_correct", False), result.get(
                "reason", "No reason provided"
            )

        except Exception as e:
            print(f"Judge error: {e}")
            return False, f"Evaluation failed: {e}"

    def evaluate_batch(self, questions: list[dict[str, Any]]) -> EvaluationReport:
        samples: list[EvaluationSample] = []

        for i, q in enumerate(questions, 1):
            print(f"Evaluating {i}/{len(questions)}: {q['question'][:60]}...")

            # 1. Pipeline Execution
            response = self.query_engine.query(q["question"])

            # 2. Correctness Check
            is_correct, reason = self.evaluate_answer(
                q["question"], response.response, q["ground_truth"]
            )

            print(f"  -> {'✅' if is_correct else '❌'}  {reason[:80]}...")

            # 3. Create Sample
            sample = EvaluationSample(
                question=q["question"],
                generated_answer=response.response,
                ground_truth=q["ground_truth"],
                retrieved_contexts=[n.get_content() for n in response.source_nodes],
                timing=TimingMetrics(
                    retrieval_seconds=response.metadata.get("retrieval_seconds", 0.0),
                    synthesis_seconds=response.metadata.get("synthesis_seconds", 0.0),
                    total_seconds=response.metadata.get("total_seconds", 0.0),
                ),
                is_correct=is_correct,
                faithfulness=1.0 if is_correct else 0.0,
                answer_relevancy=1.0 if is_correct else 0.0,
            )
            samples.append(sample)

        return self._create_report(samples)

    def _create_report(self, samples: list[EvaluationSample]) -> EvaluationReport:
        if not samples:
            # Return empty report
            return EvaluationReport(
                model=settings.groq.model,
                total_samples=0,
                avg_retrieval_seconds=0,
                avg_synthesis_seconds=0,
                avg_total_seconds=0,
                samples=[],
            )

        avg_retrieval = sum(s.timing.retrieval_seconds for s in samples) / len(samples)
        avg_synthesis = sum(s.timing.synthesis_seconds for s in samples) / len(samples)
        avg_total = sum(s.timing.total_seconds for s in samples) / len(samples)
        accuracy = sum(1 for s in samples if s.is_correct) / len(samples)

        print("\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.0%}")
        print(f"  Avg Latency: {avg_total:.2f}s")

        return EvaluationReport(
            model=settings.groq.model,
            total_samples=len(samples),
            avg_retrieval_seconds=round(avg_retrieval, 4),
            avg_synthesis_seconds=round(avg_synthesis, 4),
            avg_total_seconds=round(avg_total, 4),
            avg_faithfulness=round(accuracy, 4),
            samples=samples,
        )

    def save_report(self, report: EvaluationReport) -> None:
        """Save report to JSON."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"custom_eval_report_{timestamp}.json"

        output_dir = settings.BASE_DIR / "reports"
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\nReport saved to: {filepath}")
