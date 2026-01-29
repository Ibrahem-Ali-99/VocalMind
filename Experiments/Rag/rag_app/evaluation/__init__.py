"""
Evaluation module for RAG system.

Provides RAGAS-based metrics.
"""

from rag_app.evaluation.ragas_evaluator import RAGASEvaluator
from rag_app.evaluation.models import EvaluationSample, EvaluationReport

__all__ = [
    "RAGASEvaluator",
    "EvaluationSample",
    "EvaluationReport",
]
