"""
Pydantic models for evaluation results.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TimingMetrics(BaseModel):
    """Timing information for a single query."""

    retrieval_seconds: float = Field(description="Time spent on retrieval")
    synthesis_seconds: float = Field(description="Time spent on LLM synthesis")
    total_seconds: float = Field(description="Total query time")


class EvaluationSample(BaseModel):
    """A single evaluation sample with question, answer, and metrics."""

    question: str
    generated_answer: str
    ground_truth: list[str] | str | None = None
    retrieved_contexts: list[str] = Field(default_factory=list)
    timing: TimingMetrics

    # RAGAS metrics (optional, filled by RAGASEvaluator)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None

    # HERB-specific (optional)
    product: str | None = None
    question_type: str | None = None
    citations: list[str] = Field(default_factory=list)
    is_correct: bool | None = None


class EvaluationReport(BaseModel):
    """Aggregated evaluation report."""

    timestamp: datetime = Field(default_factory=datetime.now)
    model: str
    total_samples: int

    # Aggregate timing
    avg_retrieval_seconds: float
    avg_synthesis_seconds: float
    avg_total_seconds: float

    # Aggregate RAGAS metrics
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    avg_context_recall: float | None = None

    # HERB-specific
    accuracy: float | None = None
    samples: list[EvaluationSample] = Field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
