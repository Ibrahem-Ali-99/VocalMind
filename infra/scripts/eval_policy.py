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
    precision_recall_f1,
    write_json,
)


def evaluate_policy(
    gold_path: Path | None = None,
    predictions_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    gold_data = load_json(gold_path or (EXPECTED_DIR / "policy_gold.json"))
    predictions_doc = load_json(predictions_path or BASELINE_PREDICTIONS_PATH)
    predicted = predictions_doc.get("policy", {})
    thresholds = load_json(THRESHOLDS_PATH).get("policy", {})
    prediction_map = {item["id"]: item for item in predicted.get("samples", [])}

    tp = fp = fn = 0
    attribution_total = 0
    attribution_correct = 0
    sample_reports: list[dict[str, Any]] = []

    for sample in gold_data.get("samples", []):
        sample_id = sample["id"]
        pred = prediction_map.get(sample_id, {})
        gold_violations = set(sample.get("violations", []))
        pred_violations = set(pred.get("violations", []))
        gold_policy_ids = set(sample.get("violated_policy_ids", []))
        pred_policy_ids = set(pred.get("violated_policy_ids", []))

        tp += len(gold_violations.intersection(pred_violations))
        fp += len(pred_violations - gold_violations)
        fn += len(gold_violations - pred_violations)

        attribution_total += len(gold_policy_ids)
        attribution_correct += len(gold_policy_ids.intersection(pred_policy_ids))

        sample_reports.append(
            {
                "id": sample_id,
                "gold_violations": sorted(gold_violations),
                "pred_violations": sorted(pred_violations),
                "gold_policy_ids": sorted(gold_policy_ids),
                "pred_policy_ids": sorted(pred_policy_ids),
            }
        )

    precision, recall, _ = precision_recall_f1(tp, fp, fn)
    attribution_accuracy = (attribution_correct / attribution_total) if attribution_total else 1.0

    min_precision = float(thresholds.get("min_violation_precision", 0.0))
    min_recall = float(thresholds.get("min_violation_recall", 0.0))
    min_attr = float(thresholds.get("min_policy_attribution_accuracy", 0.0))
    report = {
        "component": "policy",
        "metrics": {
            "violation_precision": precision,
            "violation_recall": recall,
            "policy_attribution_accuracy": attribution_accuracy,
        },
        "thresholds": thresholds,
        "passed": precision >= min_precision and recall >= min_recall and attribution_accuracy >= min_attr,
        "samples": sample_reports,
    }
    write_json(report_path or (REPORTS_DIR / "policy_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy violation and attribution quality.")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_policy(args.gold, args.predictions, args.report)
    print(
        f"[policy] passed={report['passed']} precision={report['metrics']['violation_precision']:.4f} "
        f"recall={report['metrics']['violation_recall']:.4f} attribution={report['metrics']['policy_attribution_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
