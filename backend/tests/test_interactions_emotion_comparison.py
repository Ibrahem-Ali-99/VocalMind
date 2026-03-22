from types import SimpleNamespace

from app.api.routes.interactions import _build_emotion_comparison_payload


def test_emotion_comparison_payload_compact_distribution_and_quality():
    utterances = [
        SimpleNamespace(text="Thanks a lot", emotion="happy", emotion_confidence=0.9),
        SimpleNamespace(text="This is unacceptable", emotion="neutral", emotion_confidence=0.7),
        SimpleNamespace(text="I am still upset", emotion="frustrated", emotion_confidence=0.8),
    ]

    payload = _build_emotion_comparison_payload(utterances)

    assert payload["totalUtterances"] == 3
    assert "acoustic" in payload["distributions"]
    assert "text" in payload["distributions"]
    assert "fused" in payload["distributions"]

    acoustic_rows = payload["distributions"]["acoustic"]
    assert any(row["emotion"] == "happy" for row in acoustic_rows)

    quality = payload["quality"]
    assert 0.0 <= quality["acousticTextAgreementRate"] <= 100.0
    assert 0.0 <= quality["fusedMatchesAcousticRate"] <= 100.0
    assert 0.0 <= quality["fusedMatchesTextRate"] <= 100.0
    assert quality["disagreementCount"] >= 0


def test_emotion_comparison_payload_handles_empty_utterances():
    payload = _build_emotion_comparison_payload([])

    assert payload["totalUtterances"] == 0
    assert payload["distributions"]["acoustic"] == []
    assert payload["distributions"]["text"] == []
    assert payload["distributions"]["fused"] == []
    assert payload["quality"]["disagreementCount"] == 0
