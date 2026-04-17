from pathlib import Path
from typing import Any

from app.core.emotion_fusion import build_deterministic_emotion_analysis


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3")


def is_supported_audio_filename(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def audio_content_type(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    return "audio/wav"


def normalize_emotion_label(label: str | None) -> str:
    if not label:
        return "unknown"

    value = label.split("/")[-1].strip().lower() if "/" in label else label.strip().lower()
    if not value:
        return "unknown"

    alias_map = {
        "joy": "happy",
        "happiness": "happy",
        "happy": "happy",
        "anger": "angry",
        "angry": "angry",
        "sadness": "sad",
        "sad": "sad",
        "fear": "frustrated",
        "fearful": "frustrated",
        "frustration": "frustrated",
        "frustrated": "frustrated",
        "disgust": "frustrated",
        "disgusted": "frustrated",
        "surprise": "surprised",
        "surprised": "surprised",
        "neutral": "neutral",
        "calm": "neutral",
        "other": "unknown",
        "<unk>": "unknown",
        "unknown": "unknown",
    }
    return alias_map.get(value, value)


def normalize_emotion_scores(emotions: list[dict[str, Any]] | None) -> list[dict[str, float | str]]:
    normalized: list[dict[str, float | str]] = []
    for item in emotions or []:
        label = normalize_emotion_label(item.get("label"))
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        normalized.append({"label": label, "score": score})
    return normalized


def normalize_segment_emotion_analysis(data: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue

        emotion_scores = normalize_emotion_scores(item.get("emotion_scores") or item.get("emotions"))
        emotion = normalize_emotion_label(item.get("emotion") or item.get("top_emotion") or item.get("label"))

        score_value = item.get("confidence")
        if score_value is None:
            score_value = item.get("top_score")
        if score_value is None:
            score_value = item.get("score")
        if score_value is None and emotion_scores:
            score_value = emotion_scores[0].get("score")

        try:
            confidence = float(score_value or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        if emotion == "unknown" and emotion_scores:
            emotion = normalize_emotion_label(str(emotion_scores[0].get("label")))

        if not emotion_scores and emotion != "unknown":
            emotion_scores = [{"label": emotion, "score": confidence}]

        normalized.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "emotion_scores": emotion_scores,
            }
        )

    return normalized


def normalize_emotion_analysis(data: dict[str, Any]) -> dict[str, Any]:
    segment_emotions = normalize_segment_emotion_analysis(
        data.get("segment_emotions") or data.get("segments")
    )

    if "top_emotion" in data or "emotions" in data:
        emotions = normalize_emotion_scores(data.get("emotions"))
        top_emotion = normalize_emotion_label(data.get("top_emotion"))
        top_score = data.get("top_score")
        if top_score is None and emotions:
            top_score = emotions[0]["score"]
        return {
            "top_emotion": top_emotion,
            "top_score": float(top_score or 0.0),
            "emotions": emotions,
            "segment_emotions": segment_emotions,
        }

    raw = data.get("raw_result", {})
    labels = raw.get("labels", [])
    scores = raw.get("scores", [])
    emotions = normalize_emotion_scores(
        [{"label": label, "score": score} for label, score in zip(labels, scores)]
    )
    top_emotion = normalize_emotion_label(data.get("emotion"))
    top_score = float(data.get("confidence", 0.0) or 0.0)
    if not emotions and top_emotion != "unknown":
        emotions = [{"label": top_emotion, "score": top_score}]
    return {
        "top_emotion": top_emotion,
        "top_score": top_score,
        "emotions": emotions,
        "segment_emotions": segment_emotions,
    }


def normalize_transcription_response(data: dict[str, Any]) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for item in data.get("segments", []):
        segment = {
            "start": float(item.get("start", 0.0) or 0.0),
            "end": float(item.get("end", 0.0) or 0.0),
            "text": (item.get("text") or "").strip(),
        }
        if "speaker" in item:
            segment["speaker"] = item.get("speaker") or "UNKNOWN"
        if "overlap" in item:
            segment["overlap"] = bool(item.get("overlap"))
        if "speaker_meta" in item and isinstance(item.get("speaker_meta"), dict):
            segment["speaker_meta"] = dict(item.get("speaker_meta") or {})
        segments.append(segment)

    text = (data.get("text") or "").strip()
    if not text:
        text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()

    return {
        "text": text,
        "language": data.get("language") or "",
        "segments": segments,
    }


def normalize_diarization_response(data: dict[str, Any]) -> dict[str, Any]:
    raw_segments = data.get("segments", [])
    diarized_segments: list[dict[str, Any]] = []
    for item in raw_segments:
        if "speaker" not in item:
            continue
        diarized_segments.append(
            {
                "start": float(item.get("start", item.get("start_time", 0.0)) or 0.0),
                "end": float(item.get("end", item.get("end_time", 0.0)) or 0.0),
                "speaker": item.get("speaker") or "UNKNOWN",
            }
        )
    return {"segments": diarized_segments}


def normalize_vad_response(data: dict[str, Any]) -> dict[str, Any]:
    if "speech_segments" in data:
        segments = data.get("speech_segments", [])
    else:
        segments = [
            {
                "start": item.get("start_time", item.get("start", 0.0)),
                "end": item.get("end_time", item.get("end", 0.0)),
            }
            for item in data.get("segments", [])
        ]

    normalized = [
        {
            "start": float(item.get("start", 0.0) or 0.0),
            "end": float(item.get("end", 0.0) or 0.0),
        }
        for item in segments
    ]
    return {"speech_segments": normalized}


def normalize_full_response(data: dict[str, Any]) -> dict[str, Any]:
    emotions = normalize_emotion_scores(data.get("emotions"))
    top_emotion = normalize_emotion_label(data.get("top_emotion"))
    top_score = data.get("top_score")
    if top_score is None and emotions:
        top_score = emotions[0]["score"]

    segments: list[dict[str, Any]] = []
    for item in data.get("segments", []):
        emotion_scores = normalize_emotion_scores(item.get("emotion_scores"))
        emotion = normalize_emotion_label(item.get("emotion"))
        segment = {
            "start": float(item.get("start", 0.0) or 0.0),
            "end": float(item.get("end", 0.0) or 0.0),
            "text": (item.get("text") or "").strip(),
            "speaker": item.get("speaker") or "UNKNOWN",
            "emotion": emotion,
            "emotion_scores": emotion_scores,
        }
        if isinstance(item.get("speaker_meta"), dict):
            segment["speaker_meta"] = dict(item.get("speaker_meta") or {})
        segments.append(segment)

    text = (data.get("text") or "").strip()
    if not text:
        text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()

    return {
        "text": text,
        "language": data.get("language") or "",
        "segments": segments,
        "top_emotion": top_emotion,
        "top_score": float(top_score or 0.0),
        "emotions": emotions,
    }


def build_local_full_response(
    transcription: dict[str, Any], emotion_analysis: dict[str, Any]
) -> dict[str, Any]:
    normalized_transcription = normalize_transcription_response(transcription)
    normalized_emotion = normalize_emotion_analysis(emotion_analysis)
    segment_emotions = normalized_emotion.get("segment_emotions") or []

    segments = []
    for index, item in enumerate(normalized_transcription["segments"]):
        segment_emotion_data = segment_emotions[index] if index < len(segment_emotions) else {}
        segment_emotion = normalize_emotion_label(segment_emotion_data.get("emotion"))
        segment_scores = normalize_emotion_scores(segment_emotion_data.get("emotion_scores"))

        # Local emotion service can return clip-level emotion only; derive per-turn labels from text.
        if segment_emotion == "unknown" or not segment_scores:
            text_fallback = build_deterministic_emotion_analysis(item["text"])
            fallback_emotion = normalize_emotion_label(text_fallback.get("top_emotion"))
            fallback_scores = normalize_emotion_scores(text_fallback.get("emotions"))

            if segment_emotion == "unknown" and fallback_emotion != "unknown":
                segment_emotion = fallback_emotion
            if not segment_scores and fallback_scores:
                segment_scores = fallback_scores

        if segment_emotion == "unknown":
            segment_emotion = normalized_emotion["top_emotion"]
        if not segment_scores:
            segment_scores = list(normalized_emotion["emotions"])

        segments.append(
            {
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
                "speaker": item.get("speaker", "UNKNOWN"),
                "speaker_meta": dict(item.get("speaker_meta") or {}),
                "emotion": segment_emotion,
                "emotion_scores": segment_scores,
            }
        )

    return {
        "text": normalized_transcription["text"],
        "language": normalized_transcription["language"],
        "segments": segments,
        "top_emotion": normalized_emotion["top_emotion"],
        "top_score": normalized_emotion["top_score"],
        "emotions": normalized_emotion["emotions"],
    }
