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
    safe_div,
    write_json,
)


def evaluate_resolution(
    gold_path: Path | None = None,
    predictions_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    gold_data = load_json(gold_path or (EXPECTED_DIR / "resolution_gold.json"))
    predictions_doc = load_json(predictions_path or BASELINE_PREDICTIONS_PATH)
    predicted = predictions_doc.get("resolution", {})
    thresholds = load_json(THRESHOLDS_PATH).get("resolution", {})
    prediction_map = {item["id"]: item for item in predicted.get("samples", [])}

    correct = 0
    total = 0
    step_tp = step_fp = step_fn = 0
    sample_reports: list[dict[str, Any]] = []

    for sample in gold_data.get("samples", []):
        sample_id = sample["id"]
        pred = prediction_map.get(sample_id, {})
        gold_resolved = bool(sample.get("is_resolved", False))
        pred_resolved = bool(pred.get("is_resolved", False))
        if gold_resolved == pred_resolved:
            correct += 1
        total += 1

        gold_steps = set(sample.get("missing_steps", []))
        pred_steps = set(pred.get("missing_steps", []))
        step_tp += len(gold_steps.intersection(pred_steps))
        step_fp += len(pred_steps - gold_steps)
        step_fn += len(gold_steps - pred_steps)

        sample_reports.append(
            {
                "id": sample_id,
                "gold_is_resolved": gold_resolved,
                "pred_is_resolved": pred_resolved,
                "gold_missing_steps": sorted(gold_steps),
                "pred_missing_steps": sorted(pred_steps),
            }
        )

    resolved_accuracy = safe_div(correct, total)
    _, _, missing_step_f1 = precision_recall_f1(step_tp, step_fp, step_fn)
    min_resolved = float(thresholds.get("min_resolved_accuracy", 0.0))
    min_f1 = float(thresholds.get("min_missing_step_f1", 0.0))

    report = {
        "component": "resolution",
        "metrics": {
            "resolved_accuracy": resolved_accuracy,
            "missing_step_f1": missing_step_f1,
        },
        "thresholds": thresholds,
        "passed": resolved_accuracy >= min_resolved and missing_step_f1 >= min_f1,
        "samples": sample_reports,
    }
    write_json(report_path or (REPORTS_DIR / "resolution_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate issue-resolution and missing-step quality.")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_resolution(args.gold, args.predictions, args.report)
    print(
        f"[resolution] passed={report['passed']} resolved_accuracy={report['metrics']['resolved_accuracy']:.4f} "
        f"missing_step_f1={report['metrics']['missing_step_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
