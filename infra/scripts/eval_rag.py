from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval_common import (
    BASELINE_PREDICTIONS_PATH,
    EXPECTED_DIR,
    REPORTS_DIR,
    THRESHOLDS_PATH,
    load_json,
    safe_div,
    write_json,
)


def evaluate_rag(
    gold_path: Path | None = None,
    predictions_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    gold_data = load_json(gold_path or (EXPECTED_DIR / "rag_gold.json"))
    predictions_doc = load_json(predictions_path or BASELINE_PREDICTIONS_PATH)
    predicted = predictions_doc.get("rag", {})
    thresholds = load_json(THRESHOLDS_PATH).get("rag", {})
    prediction_map = {item["id"]: item for item in predicted.get("samples", [])}

    correct = 0
    total = 0
    evidence_coverages: list[float] = []
    sample_reports: list[dict[str, Any]] = []

    for sample in gold_data.get("samples", []):
        sample_id = sample["id"]
        pred = prediction_map.get(sample_id, {})
        gold_correct = bool(sample.get("is_correct", False))
        pred_correct = bool(pred.get("is_correct", False))
        if gold_correct == pred_correct:
            correct += 1
        total += 1

        gold_evidence = set(sample.get("evidence_ids", []))
        pred_evidence = set(pred.get("evidence_ids", []))
        coverage = safe_div(len(gold_evidence.intersection(pred_evidence)), len(gold_evidence))
        evidence_coverages.append(coverage)

        sample_reports.append(
            {
                "id": sample_id,
                "gold_is_correct": gold_correct,
                "pred_is_correct": pred_correct,
                "evidence_coverage": coverage,
            }
        )

    correctness_accuracy = safe_div(correct, total)
    avg_evidence_coverage = sum(evidence_coverages) / len(evidence_coverages) if evidence_coverages else 0.0
    min_acc = float(thresholds.get("min_correctness_accuracy", 0.0))
    min_cov = float(thresholds.get("min_evidence_coverage", 0.0))

    report = {
        "component": "rag",
        "metrics": {
            "correctness_accuracy": correctness_accuracy,
            "avg_evidence_coverage": avg_evidence_coverage,
        },
        "thresholds": thresholds,
        "passed": correctness_accuracy >= min_acc and avg_evidence_coverage >= min_cov,
        "samples": sample_reports,
    }
    write_json(report_path or (REPORTS_DIR / "rag_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG correctness verdict and evidence coverage.")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_rag(args.gold, args.predictions, args.report)
    print(
        f"[rag] passed={report['passed']} correctness_accuracy={report['metrics']['correctness_accuracy']:.4f} "
        f"evidence_coverage={report['metrics']['avg_evidence_coverage']:.4f}"
    )


if __name__ == "__main__":
    main()
