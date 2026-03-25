from pathlib import Path
from typing import Any


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
    return label.split("/")[-1].strip() if "/" in label else label


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


def normalize_emotion_analysis(data: dict[str, Any]) -> dict[str, Any]:
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

    segments = []
    for item in normalized_transcription["segments"]:
        segments.append(
            {
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
                "speaker": item.get("speaker", "UNKNOWN"),
                "emotion": normalized_emotion["top_emotion"],
                "emotion_scores": normalized_emotion["emotions"],
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
