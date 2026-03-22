from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class EmotionFusionResult:
    emotion: str
    confidence: float
    text_emotion: str
    text_confidence: float
    acoustic_emotion: str
    acoustic_confidence: float
    model: str = "text_acoustic_fusion_v2"


TEXT_LEXICON: dict[str, set[str]] = {
    "happy": {"great", "good", "thanks", "thank", "helpful", "happy", "awesome", "perfect"},
    "sad": {"sad", "sorry", "unhappy", "depressed", "down"},
    "angry": {"angry", "furious", "mad", "unacceptable", "ridiculous", "terrible"},
    "frustrated": {"frustrated", "annoyed", "upset", "still", "again", "problem", "issue"},
    "neutral": set(),
}

EMOTION_NORMALIZATION: dict[str, str] = {
    "joy": "happy",
    "calm": "neutral",
    "satisfied": "happy",
    "fear": "frustrated",
    "disgust": "frustrated",
}

TEXT_LABEL_NORMALIZATION: dict[str, str] = {
    "joy": "happy",
    "happiness": "happy",
    "anger": "angry",
    "fear": "frustrated",
    "surprise": "neutral",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "disgust": "frustrated",
    "annoyance": "frustrated",
}


def _normalize_emotion(label: str) -> str:
    base = (label or "neutral").strip().lower()
    return EMOTION_NORMALIZATION.get(base, base or "neutral")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", (text or "").lower())


def infer_text_emotion(text: str) -> tuple[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return "neutral", 0.2

    scores: dict[str, int] = {label: 0 for label in TEXT_LEXICON.keys()}
    for token in tokens:
        for label, words in TEXT_LEXICON.items():
            if token in words:
                scores[label] += 1

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score == 0:
        return "neutral", 0.3

    confidence = min(0.95, 0.45 + (best_score * 0.12))
    return best_label, confidence


def _normalize_text_label(label: str) -> str:
    base = (label or "neutral").strip().lower()
    return TEXT_LABEL_NORMALIZATION.get(base, base)


@lru_cache(maxsize=1)
def _get_hf_text_classifier():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=settings.TEXT_EMOTION_MODEL,
        top_k=None,
    )


def _infer_text_emotion_hf(text: str) -> tuple[str, float]:
    classifier = _get_hf_text_classifier()
    raw = classifier(text)
    scores = raw[0] if raw and isinstance(raw[0], list) else []
    if not scores:
        return "neutral", 0.3

    top = max(scores, key=lambda item: float(item.get("score", 0.0)))
    label = _normalize_text_label(str(top.get("label", "neutral")))
    confidence = float(top.get("score", 0.3))
    return label, max(0.0, min(1.0, confidence))


def infer_text_emotion_with_provider(text: str) -> tuple[str, float]:
    provider = settings.TEXT_EMOTION_PROVIDER.strip().lower()
    if provider == "hf_transformers":
        try:
            return _infer_text_emotion_hf(text)
        except Exception as exc:
            logger.warning("HF text emotion fallback to rule-based due to error: %s", exc)

    return infer_text_emotion(text)


def fuse_emotion_signals(
    text: str,
    acoustic_emotion: str,
    acoustic_confidence: float | None = None,
) -> EmotionFusionResult:
    text_emotion, text_confidence = infer_text_emotion_with_provider(text)
    acoustic_label = _normalize_emotion(acoustic_emotion)
    acoustic_score = acoustic_confidence if acoustic_confidence is not None else 0.7
    acoustic_score = max(0.0, min(1.0, acoustic_score))

    text_weight = 0.45
    acoustic_weight = 0.55

    if text_emotion == acoustic_label:
        fused_emotion = acoustic_label
        fused_confidence = min(0.99, (text_confidence * text_weight) + (acoustic_score * acoustic_weight) + 0.08)
    else:
        # Prefer stronger signal but keep confidence conservative when modalities disagree.
        fused_emotion = text_emotion if text_confidence > acoustic_score else acoustic_label
        fused_confidence = max(0.35, (text_confidence * text_weight) + (acoustic_score * acoustic_weight) - 0.12)

    return EmotionFusionResult(
        emotion=fused_emotion,
        confidence=round(fused_confidence, 3),
        text_emotion=text_emotion,
        text_confidence=round(text_confidence, 3),
        acoustic_emotion=acoustic_label,
        acoustic_confidence=round(acoustic_score, 3),
        model=f"{settings.TEXT_EMOTION_PROVIDER}_text_acoustic_fusion_v2",
    )
