from __future__ import annotations

import ast
from pathlib import Path


def test_detect_overlaps_marks_intersecting_segments():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    detect_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "detect_overlaps"
    )
    detect_source = ast.get_source_segment(source, detect_node)
    namespace: dict[str, object] = {}
    exec(detect_source, namespace)  # noqa: S102 - controlled source from repo file
    detect_overlaps = namespace["detect_overlaps"]

    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 0.8, "end": 1.4, "speaker": "SPEAKER_01"},
        {"start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},
    ]
    result = detect_overlaps(segments)
    assert result[0]["overlap"] is True
    assert result[1]["overlap"] is True
    assert result[2]["overlap"] is False
