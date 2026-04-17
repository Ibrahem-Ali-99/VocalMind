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


def _event_indexes(labels: list[str]) -> set[int]:
    output: set[int] = set()
    previous = None
    for idx, label in enumerate(labels):
        if previous is None:
            previous = label
            continue
        if label != previous:
            output.add(idx)
            previous = label
    return output


def evaluate_emotion(
    gold_path: Path | None = None,
    predictions_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    gold_data = load_json(gold_path or (EXPECTED_DIR / "emotion_gold.json"))
    predictions_doc = load_json(predictions_path or BASELINE_PREDICTIONS_PATH)
    predicted = predictions_doc.get("emotion", {})
    thresholds = load_json(THRESHOLDS_PATH).get("emotion", {})
    prediction_map = {item["id"]: item for item in predicted.get("samples", [])}

    labels: set[str] = set()
    tp_by_label: dict[str, int] = {}
    fp_by_label: dict[str, int] = {}
    fn_by_label: dict[str, int] = {}
    shift_tp = shift_fp = shift_fn = 0
    sample_reports: list[dict[str, Any]] = []

    for sample in gold_data.get("samples", []):
        sample_id = sample["id"]
        gold_labels = sample.get("turn_emotions", [])
        pred_labels = prediction_map.get(sample_id, {}).get("turn_emotions", [])
        max_len = max(len(gold_labels), len(pred_labels))

        for idx in range(max_len):
            g = gold_labels[idx] if idx < len(gold_labels) else "__missing__"
            p = pred_labels[idx] if idx < len(pred_labels) else "__missing__"
            labels.add(g)
            labels.add(p)
            if g == p:
                tp_by_label[g] = tp_by_label.get(g, 0) + 1
            else:
                fp_by_label[p] = fp_by_label.get(p, 0) + 1
                fn_by_label[g] = fn_by_label.get(g, 0) + 1

        gold_events = _event_indexes(gold_labels)
        pred_events = _event_indexes(pred_labels)
        shift_tp += len(gold_events.intersection(pred_events))
        shift_fp += len(pred_events - gold_events)
        shift_fn += len(gold_events - pred_events)

        sample_reports.append(
            {
                "id": sample_id,
                "turn_accuracy": sum(1 for a, b in zip(gold_labels, pred_labels) if a == b) / len(gold_labels)
                if gold_labels
                else 0.0,
                "gold_shift_events": sorted(gold_events),
                "pred_shift_events": sorted(pred_events),
            }
        )

    label_f1_values: list[float] = []
    for label in labels:
        if label == "__missing__":
            continue
        precision, recall, f1 = precision_recall_f1(
            tp_by_label.get(label, 0),
            fp_by_label.get(label, 0),
            fn_by_label.get(label, 0),
        )
        label_f1_values.append(f1)

    macro_f1 = sum(label_f1_values) / len(label_f1_values) if label_f1_values else 0.0
    _, _, shift_f1 = precision_recall_f1(shift_tp, shift_fp, shift_fn)
    min_label_f1 = float(thresholds.get("min_label_f1", 0.0))
    min_shift_f1 = float(thresholds.get("min_shift_f1", 0.0))

    report = {
        "component": "emotion",
        "metrics": {
            "label_macro_f1": macro_f1,
            "shift_event_f1": shift_f1,
        },
        "thresholds": thresholds,
        "passed": macro_f1 >= min_label_f1 and shift_f1 >= min_shift_f1,
        "samples": sample_reports,
    }
    write_json(report_path or (REPORTS_DIR / "emotion_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotion turn labels and emotion-shift events.")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_emotion(args.gold, args.predictions, args.report)
    print(f"[emotion] passed={report['passed']} macro_f1={report['metrics']['label_macro_f1']:.4f} shift_f1={report['metrics']['shift_event_f1']:.4f}")


if __name__ == "__main__":
    main()
