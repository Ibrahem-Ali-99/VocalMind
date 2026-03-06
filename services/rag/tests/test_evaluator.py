"""Tests for evaluator.py — JSON parsing, result models, and prompt construction."""

import json
import os


os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from evaluator import (
    ComplianceResult,
    CorrectnessResult,
    EvaluationReport,
    _parse_json_response,
)


# ── JSON Parsing Helper ─────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json(self):
        content = '{"compliance_score": 0.85, "violations": [], "reasoning": "OK"}'
        result = _parse_json_response(content)
        assert result["compliance_score"] == 0.85
        assert result["violations"] == []

    def test_json_in_markdown_fence(self):
        content = '```json\n{"compliance_score": 0.9}\n```'
        result = _parse_json_response(content)
        assert result["compliance_score"] == 0.9

    def test_json_with_extra_text(self):
        content = 'Here is the result:\n{"correctness_score": 0.7, "is_correct": true}\nDone.'
        result = _parse_json_response(content)
        assert result["correctness_score"] == 0.7
        assert result["is_correct"] is True

    def test_invalid_json_returns_error(self):
        content = "This is not JSON at all."
        result = _parse_json_response(content)
        assert "error" in result

    def test_empty_object(self):
        result = _parse_json_response("{}")
        assert result == {}


# ── Result Models ────────────────────────────────────────────────────────────

class TestComplianceResult:
    def test_defaults(self):
        r = ComplianceResult(transcript="test transcript", compliance_score=0.0)
        assert r.compliance_score == 0.0
        assert r.violations == []
        assert r.policy_references == []
        assert r.reasoning == ""

    def test_score_bounds(self):
        r = ComplianceResult(transcript="t", compliance_score=0.5)
        assert 0.0 <= r.compliance_score <= 1.0

    def test_with_violations(self):
        r = ComplianceResult(
            transcript="t",
            compliance_score=0.3,
            violations=["Did not verify identity", "Promised unauthorized refund"],
        )
        assert len(r.violations) == 2


class TestCorrectnessResult:
    def test_defaults(self):
        r = CorrectnessResult(question="Q", agent_answer="A", correctness_score=0.0)
        assert r.is_correct is False
        assert r.reasoning == ""
        assert r.source_references == []

    def test_correct_answer(self):
        r = CorrectnessResult(
            question="Q",
            agent_answer="A",
            correctness_score=0.95,
            is_correct=True,
            reasoning="Answer matches policy.",
        )
        assert r.is_correct is True
        assert r.correctness_score == 0.95


class TestEvaluationReport:
    def test_defaults(self):
        report = EvaluationReport(eval_type="compliance")
        assert report.total_samples == 0
        assert report.avg_score == 0.0
        assert report.results == []
        assert report.timestamp  # should be auto-set

    def test_with_results(self):
        results = [
            ComplianceResult(transcript="t1", compliance_score=0.8),
            ComplianceResult(transcript="t2", compliance_score=0.6),
        ]
        report = EvaluationReport(
            eval_type="compliance",
            model="llama-3.3-70b-versatile",
            total_samples=2,
            avg_score=0.7,
            results=results,
        )
        assert report.total_samples == 2
        assert report.avg_score == 0.7

    def test_json_serialization(self):
        report = EvaluationReport(eval_type="correctness", total_samples=0)
        data = json.loads(report.model_dump_json())
        assert data["eval_type"] == "correctness"
