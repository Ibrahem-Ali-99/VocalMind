from __future__ import annotations

import importlib
from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    _module = importlib.import_module("speaker_role_classifier")
except Exception as exc:  # pragma: no cover - environment specific
    pytest.skip(f"speaker_role_classifier dependencies unavailable: {exc}", allow_module_level=True)

SpeakerRoleClassifier = _module.SpeakerRoleClassifier
_normalize_speaker_label = _module._normalize_speaker_label


def test_normalize_speaker_label_uses_token_matching():
    assert _normalize_speaker_label("customer") == "customer"
    assert _normalize_speaker_label("agent") == "agent"
    assert _normalize_speaker_label("not_customer_service") is None


def test_relabel_segments_preserves_diarization_when_model_unavailable(tmp_path):
    classifier = SpeakerRoleClassifier(model_dir=Path(tmp_path / "missing"), enabled=True)
    segments = [
        {"text": "Hi", "speaker": "SPEAKER_00"},
        {"text": "How can I help you today?", "speaker": "SPEAKER_01"},
    ]

    output = classifier.relabel_segments(segments)
    assert output[0]["speaker"] == "SPEAKER_00"
    assert output[1]["speaker"] == "SPEAKER_01"
    assert output[0]["speaker_meta"]["source"] == "diarization"
    assert output[1]["speaker_meta"]["fallback_reason"] == "classifier_unavailable"
