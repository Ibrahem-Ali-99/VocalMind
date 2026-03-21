from app.core.emotion_fusion import _normalize_text_label, fuse_emotion_signals


def test_fuse_emotion_signals_prefers_agreement():
    result = fuse_emotion_signals(
        text="Thanks, this was great and very helpful.",
        acoustic_emotion="happy",
        acoustic_confidence=0.8,
    )
    assert result.emotion == "happy"
    assert 0.0 <= result.confidence <= 1.0
    assert result.model.endswith("_text_acoustic_fusion_v2")


def test_fuse_emotion_signals_handles_disagreement():
    result = fuse_emotion_signals(
        text="This is unacceptable and terrible.",
        acoustic_emotion="calm",
        acoustic_confidence=0.6,
    )
    assert result.emotion in {"angry", "frustrated", "neutral"}
    assert 0.0 <= result.confidence <= 1.0


def test_text_label_normalization_maps_supported_labels():
    assert _normalize_text_label("joy") == "happy"
    assert _normalize_text_label("anger") == "angry"
    assert _normalize_text_label("sadness") == "sad"
