from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "infra" / "benchmarks"
EXPECTED_DIR = BENCHMARK_DIR / "expected"
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
REPORTS_DIR = BENCHMARK_DIR / "reports"
THRESHOLDS_PATH = BENCHMARK_DIR / "schema" / "thresholds.json"
BASELINE_PREDICTIONS_PATH = FIXTURES_DIR / "baseline_predictions.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, (2 * precision * recall) / (precision + recall)
