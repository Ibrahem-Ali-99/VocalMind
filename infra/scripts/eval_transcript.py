from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from eval_common import (
    BASELINE_PREDICTIONS_PATH,
    EXPECTED_DIR,
    REPORTS_DIR,
    THRESHOLDS_PATH,
    load_json,
    write_json,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def evaluate_transcript(
    gold_path: Path | None = None,
    predictions_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    gold_data = load_json(gold_path or (EXPECTED_DIR / "transcript_gold.json"))
    predictions_doc = load_json(predictions_path or BASELINE_PREDICTIONS_PATH)
    predicted = predictions_doc.get("transcript", {})
    thresholds = load_json(THRESHOLDS_PATH).get("transcript", {})
    prediction_map = {item["id"]: item for item in predicted.get("samples", [])}

    total_tokens = 0
    total_edits = 0
    speaker_total = 0
    speaker_correct = 0
    confusion: dict[str, dict[str, int]] = {"agent": {"agent": 0, "customer": 0}, "customer": {"agent": 0, "customer": 0}}

    sample_reports: list[dict[str, Any]] = []
    for sample in gold_data.get("samples", []):
        sample_id = sample["id"]
        pred = prediction_map.get(sample_id, {"text": "", "turns": []})

        ref_tokens = _tokenize(sample.get("text", ""))
        hyp_tokens = _tokenize(pred.get("text", ""))
        edits = _levenshtein(ref_tokens, hyp_tokens)
        total_tokens += len(ref_tokens)
        total_edits += edits

        ref_turns = sample.get("turns", [])
        hyp_turns = pred.get("turns", [])
        turn_len = min(len(ref_turns), len(hyp_turns))
        local_total = 0
        local_correct = 0
        for idx in range(turn_len):
            ref_speaker = ref_turns[idx].get("speaker", "customer")
            hyp_speaker = hyp_turns[idx].get("speaker", "customer")
            if ref_speaker not in confusion:
                confusion[ref_speaker] = {"agent": 0, "customer": 0}
            if hyp_speaker not in confusion[ref_speaker]:
                confusion[ref_speaker][hyp_speaker] = 0
            confusion[ref_speaker][hyp_speaker] += 1
            local_total += 1
            if ref_speaker == hyp_speaker:
                local_correct += 1

        speaker_total += local_total
        speaker_correct += local_correct
        sample_reports.append(
            {
                "id": sample_id,
                "wer": edits / len(ref_tokens) if ref_tokens else 0.0,
                "speaker_accuracy": local_correct / local_total if local_total else 0.0,
            }
        )

    wer = total_edits / total_tokens if total_tokens else 0.0
    speaker_accuracy = speaker_correct / speaker_total if speaker_total else 0.0
    max_wer = float(thresholds.get("max_wer", 1.0))
    min_speaker_accuracy = float(thresholds.get("min_speaker_accuracy", 0.0))

    report = {
        "component": "transcript",
        "metrics": {
            "wer": wer,
            "speaker_accuracy": speaker_accuracy,
            "confusion": confusion,
        },
        "thresholds": thresholds,
        "passed": wer <= max_wer and speaker_accuracy >= min_speaker_accuracy,
        "samples": sample_reports,
    }
    write_json(report_path or (REPORTS_DIR / "transcript_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transcript and speaker-role quality.")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_transcript(args.gold, args.predictions, args.report)
    print(f"[transcript] passed={report['passed']} wer={report['metrics']['wer']:.4f} speaker_accuracy={report['metrics']['speaker_accuracy']:.4f}")


if __name__ == "__main__":
    main()
