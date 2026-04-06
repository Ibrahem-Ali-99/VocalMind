"""
VocalMind Final RAG — Dual Evaluators.

Two evaluation modes for customer-service call analysis:

1. **PolicyComplianceEvaluator**
   Retrieves full policy sections (parents) and checks whether an agent's
   interaction complies with company policies.  Returns a compliance score
   (0-1), list of violations, and policy references.

2. **AnswerCorrectnessEvaluator**
   Retrieves precise knowledge snippets (children) and evaluates whether
   a specific agent answer is factually correct.  Returns a correctness
   score (0-1), reasoning, and source references.

Both evaluators use Groq LLM-as-a-Judge.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from .config import settings
    from .query_engine import RAGQueryEngine
except ImportError:  # pragma: no cover - allows direct script/test imports
    from config import settings
    from query_engine import RAGQueryEngine


# ── Result Models ─────────────────────────────────────────────────────────────


class ComplianceResult(BaseModel):
    """Result of a policy compliance check."""

    transcript: str
    compliance_score: float = Field(ge=0.0, le=1.0, description="0 = non-compliant, 1 = fully compliant")
    violations: list[str] = Field(default_factory=list)
    policy_references: list[str] = Field(default_factory=list)
    reasoning: str = ""
    retrieved_policies: list[dict] = Field(default_factory=list)
    retrieval_seconds: float = 0.0
    evaluation_seconds: float = 0.0


class CorrectnessResult(BaseModel):
    """Result of an answer correctness check."""

    question: str
    agent_answer: str
    correctness_score: float = Field(ge=0.0, le=1.0, description="0 = incorrect, 1 = correct")
    is_correct: bool = False
    reasoning: str = ""
    source_references: list[str] = Field(default_factory=list)
    retrieved_snippets: list[dict] = Field(default_factory=list)
    retrieval_seconds: float = 0.0
    evaluation_seconds: float = 0.0


class EvaluationReport(BaseModel):
    """Aggregated report for a batch of evaluations."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    eval_type: str  # "compliance" or "correctness"
    model: str = ""
    total_samples: int = 0
    avg_score: float = 0.0
    results: list[ComplianceResult | CorrectnessResult] = Field(default_factory=list)


# ── Compliance Evaluator ──────────────────────────────────────────────────────

COMPLIANCE_PROMPT = """\
You are a compliance auditor for a customer-service organization.

You are given:
1. A transcript of an agent's interaction with a customer.
2. Relevant company policy sections retrieved from the policy database.

Evaluate whether the agent's actions and statements comply with the company policies.

--- COMPANY POLICIES ---
{policies}

--- AGENT TRANSCRIPT ---
{transcript}

--- INSTRUCTIONS ---
Analyze the transcript against each relevant policy section.  Identify any
violations (specific actions or statements that break policy rules).  Also
note any commendable compliance.

Respond with ONLY valid JSON — no markdown, no explanation outside JSON:
{{
    "compliance_score": <float 0.0 to 1.0>,
    "violations": ["violation 1", "violation 2", ...],
    "policy_references": ["Policy Section X: Title", ...],
    "reasoning": "Brief explanation of your assessment"
}}
"""


class PolicyComplianceEvaluator:
    """Evaluates agent transcripts against company policy (parents collection)."""

    def __init__(self, engine: RAGQueryEngine | None = None) -> None:
        self.engine = engine or RAGQueryEngine()

    def check(
        self,
        transcript: str,
        org_filter: str | None = None,
        verbose: bool = False,
    ) -> ComplianceResult:
        """
        Run a policy-compliance check on a transcript.

        Args:
            transcript:  The agent's call transcript text.
            org_filter:  Filter retrieval to a specific organization.
            verbose:     Print intermediate results.
        """
        import time

        # 1. Retrieve relevant policy sections (parents)
        result = self.engine.query_compliance(transcript, org_filter=org_filter, verbose=verbose)
        retrieval_time = result["timing"]["retrieval"]

        policies_text = "\n\n---\n\n".join(
            f"[{c['metadata'].get('doc_id', 'N/A')} | "
            f"{c['metadata'].get('Header 1', '')} > {c['metadata'].get('Header 2', '')}]\n"
            f"{c['text']}"
            for c in result["chunks"]
        )

        if not policies_text.strip():
            return ComplianceResult(
                transcript=transcript,
                compliance_score=0.5,
                reasoning="No relevant policies found for this transcript.",
                retrieval_seconds=retrieval_time,
            )

        # 2. LLM-as-Judge
        prompt = COMPLIANCE_PROMPT.format(policies=policies_text, transcript=transcript)

        t0 = time.perf_counter()
        response = self.engine.groq_client.chat.completions.create(
            model=settings.groq.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        eval_time = time.perf_counter() - t0

        # 3. Parse response
        content = response.choices[0].message.content.strip()
        parsed = _parse_json_response(content)

        return ComplianceResult(
            transcript=transcript,
            compliance_score=float(parsed.get("compliance_score", 0.5)),
            violations=parsed.get("violations", []),
            policy_references=parsed.get("policy_references", []),
            reasoning=parsed.get("reasoning", ""),
            retrieved_policies=[
                {"text": c["text"][:200], "metadata": c["metadata"]}
                for c in result["chunks"]
            ],
            retrieval_seconds=retrieval_time,
            evaluation_seconds=round(eval_time, 4),
        )


# ── Answer Correctness Evaluator ──────────────────────────────────────────────

CORRECTNESS_PROMPT = """\
You are an impartial quality auditor for a customer-service call center.

You are given:
1. A customer's question.
2. The agent's answer during the call.
3. Relevant knowledge base snippets retrieved from the company's documents.

Determine whether the agent's answer is **factually correct** based on the
company's official knowledge base.

--- KNOWLEDGE BASE SNIPPETS ---
{snippets}

--- CUSTOMER QUESTION ---
{question}

--- AGENT ANSWER ---
{answer}

--- INSTRUCTIONS ---
Compare the agent's answer against the knowledge base snippets.
- If the answer is accurate and consistent with the knowledge base → high score.
- If the answer contains factual errors or contradicts the knowledge base → low score.
- If the knowledge base doesn't cover the topic → note it and give a middle score.

Respond with ONLY valid JSON — no markdown, no explanation outside JSON:
{{
    "correctness_score": <float 0.0 to 1.0>,
    "is_correct": <boolean>,
    "reasoning": "Brief explanation",
    "source_references": ["Reference 1", "Reference 2", ...]
}}
"""


class AnswerCorrectnessEvaluator:
    """Evaluates agent answers against company knowledge base (children collection)."""

    def __init__(self, engine: RAGQueryEngine | None = None) -> None:
        self.engine = engine or RAGQueryEngine()

    def check(
        self,
        question: str,
        agent_answer: str,
        org_filter: str | None = None,
        verbose: bool = False,
    ) -> CorrectnessResult:
        """
        Check if an agent's answer is factually correct.

        Args:
            question:     The customer's question.
            agent_answer: The agent's response.
            org_filter:   Filter retrieval to a specific organization.
            verbose:      Print intermediate results.
        """
        import time

        # 1. Retrieve relevant snippets (children)
        result = self.engine.query_answer(question, org_filter=org_filter, verbose=verbose)
        retrieval_time = result["timing"]["retrieval"]

        snippets_text = "\n\n---\n\n".join(
            f"[{c['metadata'].get('doc_id', 'N/A')} | "
            f"{c['metadata'].get('source_file', '')}]\n"
            f"{c['text']}"
            for c in result["chunks"]
        )

        if not snippets_text.strip():
            return CorrectnessResult(
                question=question,
                agent_answer=agent_answer,
                correctness_score=0.5,
                reasoning="No relevant knowledge base entries found for this question.",
                retrieval_seconds=retrieval_time,
            )

        # 2. LLM-as-Judge
        prompt = CORRECTNESS_PROMPT.format(
            snippets=snippets_text, question=question, answer=agent_answer
        )

        t0 = time.perf_counter()
        response = self.engine.groq_client.chat.completions.create(
            model=settings.groq.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        eval_time = time.perf_counter() - t0

        # 3. Parse response
        content = response.choices[0].message.content.strip()
        parsed = _parse_json_response(content)

        return CorrectnessResult(
            question=question,
            agent_answer=agent_answer,
            correctness_score=float(parsed.get("correctness_score", 0.5)),
            is_correct=bool(parsed.get("is_correct", False)),
            reasoning=parsed.get("reasoning", ""),
            source_references=parsed.get("source_references", []),
            retrieved_snippets=[
                {"text": c["text"][:200], "metadata": c["metadata"]}
                for c in result["chunks"]
            ],
            retrieval_seconds=retrieval_time,
            evaluation_seconds=round(eval_time, 4),
        )


# ── Batch Evaluation ──────────────────────────────────────────────────────────


def run_compliance_batch(
    transcripts: list[str],
    org_filter: str | None = None,
    verbose: bool = False,
) -> EvaluationReport:
    """Evaluate a batch of transcripts for policy compliance."""
    evaluator = PolicyComplianceEvaluator()
    results: list[ComplianceResult] = []

    for i, transcript in enumerate(transcripts, 1):
        print(f"\n[{i}/{len(transcripts)}] Checking compliance...")
        result = evaluator.check(transcript, org_filter=org_filter, verbose=verbose)
        results.append(result)
        status = "PASS" if result.compliance_score >= 0.7 else "FAIL"
        print(f"  Score: {result.compliance_score:.2f} [{status}]")
        if result.violations:
            for v in result.violations:
                print(f"  Violation: {v}")

    avg_score = sum(r.compliance_score for r in results) / len(results) if results else 0
    report = EvaluationReport(
        eval_type="compliance",
        model=settings.groq.model,
        total_samples=len(results),
        avg_score=round(avg_score, 4),
        results=results,
    )
    _save_report(report, "compliance")
    return report


def run_correctness_batch(
    qa_pairs: list[dict[str, str]],
    org_filter: str | None = None,
    verbose: bool = False,
) -> EvaluationReport:
    """
    Evaluate a batch of Q&A pairs for answer correctness.

    Args:
        qa_pairs: List of dicts with 'question' and 'answer' keys.
    """
    evaluator = AnswerCorrectnessEvaluator()
    results: list[CorrectnessResult] = []

    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n[{i}/{len(qa_pairs)}] Checking answer: {qa['question'][:60]}...")
        result = evaluator.check(
            qa["question"], qa["answer"], org_filter=org_filter, verbose=verbose
        )
        results.append(result)
        icon = "correct" if result.is_correct else "incorrect"
        print(f"  Score: {result.correctness_score:.2f} [{icon}]  {result.reasoning[:80]}")

    avg_score = sum(r.correctness_score for r in results) / len(results) if results else 0
    report = EvaluationReport(
        eval_type="correctness",
        model=settings.groq.model,
        total_samples=len(results),
        avg_score=round(avg_score, 4),
        results=results,
    )
    _save_report(report, "correctness")
    return report


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences if present
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            clean = part.replace("json", "", 1).strip() if part.strip().startswith("json") else part.strip()
            if clean.startswith("{"):
                content = clean
                break

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find embedded JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        return {"error": f"Failed to parse LLM response: {content[:200]}"}


def _save_report(report: EvaluationReport, prefix: str) -> Path:
    """Save evaluation report to JSON."""
    reports_dir = settings.BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"{prefix}_report_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"\n  Report saved → {path}")
    return path
