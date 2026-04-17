from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval_common import BASELINE_PREDICTIONS_PATH, REPORTS_DIR, write_json
from eval_emotion import evaluate_emotion
from eval_policy import evaluate_policy
from eval_rag import evaluate_rag
from eval_resolution import evaluate_resolution
from eval_transcript import evaluate_transcript


def main() -> None:
    reports = [
        evaluate_transcript(predictions_path=BASELINE_PREDICTIONS_PATH),
        evaluate_emotion(predictions_path=BASELINE_PREDICTIONS_PATH),
        evaluate_policy(predictions_path=BASELINE_PREDICTIONS_PATH),
        evaluate_rag(predictions_path=BASELINE_PREDICTIONS_PATH),
        evaluate_resolution(predictions_path=BASELINE_PREDICTIONS_PATH),
    ]

    aggregate = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": all(report.get("passed") for report in reports),
        "components": {
            report["component"]: {
                "passed": report["passed"],
                "metrics": report["metrics"],
            }
            for report in reports
        },
    }
    out = REPORTS_DIR / "aggregate_report.json"
    write_json(out, aggregate)
    print(json.dumps(aggregate, indent=2))
    if not aggregate["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
