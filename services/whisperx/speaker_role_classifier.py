from __future__ import annotations

import logging
import inspect
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)

_CUSTOMER_LABEL_ALIASES = {
    "customer",
    "client",
    "caller",
    "user",
    "speaker_0",
    "speaker0",
    "s0",
}
_AGENT_LABEL_ALIASES = {
    "agent",
    "support",
    "representative",
    "advisor",
    "speaker_1",
    "speaker1",
    "s1",
}
_AGENT_TEXT_CUES = (
    "how can i help",
    "let me check",
    "can you provide",
    "i have reset",
    "i've reset",
    "check your email",
    "what's the issue",
    "sure, i can help",
)
_CUSTOMER_TEXT_CUES = (
    "i need help",
    "i can't access",
    "cannot access",
    "my account",
    "it's ",
    "thank you",
    "thanks",
    "thank you so much",
)


class SpeakerRoleClassifier:
    """Optional text classifier that maps transcript segments to speaker roles."""

    def __init__(
        self,
        model_dir: Path,
        enabled: bool = True,
        min_confidence: float = 0.68,
        min_text_characters: int = 12,
    ) -> None:
        self._model_dir = model_dir
        self._enabled = enabled
        self._min_confidence = min_confidence
        self._min_text_characters = min_text_characters
        self._tokenizer = None
        self._model = None
        self._failed = False

    @property
    def is_available(self) -> bool:
        return self._enabled and not self._failed and self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if not self._enabled or self._failed:
            return
        if self._model is not None and self._tokenizer is not None:
            return
        if not self._model_dir.exists():
            logger.info("Speaker role model not found at %s; using diarization labels only.", self._model_dir)
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self._model_dir)).to("cpu").eval()
            logger.info("Loaded speaker role classifier from %s", self._model_dir)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._failed = True
            self._model = None
            self._tokenizer = None
            logger.warning("Failed to load speaker role classifier: %s", exc)

    def relabel_segments(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return segments

        self.load()
        speaker_votes: dict[str, dict[str, float]] = {}

        if not self.is_available:
            for segment in segments:
                original_speaker = (segment.get("speaker") or "UNKNOWN").strip() or "UNKNOWN"
                text = (segment.get("text") or "").strip()
                cue_label = _label_from_text_cues(text)
                if cue_label:
                    segment["speaker"] = cue_label
                    _set_speaker_meta(
                        segment,
                        source="text_cue",
                        confidence=0.99,
                        raw_label="cue_rule",
                        diarization_speaker=original_speaker,
                        fallback_reason="classifier_unavailable",
                    )
                else:
                    _set_speaker_meta(
                        segment,
                        source="diarization",
                        raw_label=original_speaker,
                        diarization_speaker=original_speaker,
                        fallback_reason="classifier_unavailable",
                    )
            return segments

        for segment in segments:
            original_speaker = (segment.get("speaker") or "UNKNOWN").strip() or "UNKNOWN"
            text = (segment.get("text") or "").strip()
            if not text:
                _set_speaker_meta(
                    segment,
                    source="diarization",
                    confidence=1.0,
                    raw_label=original_speaker,
                    diarization_speaker=original_speaker,
                    fallback_reason="empty_text",
                )
                continue

            cue_label = _label_from_text_cues(text)
            if cue_label:
                segment["speaker"] = cue_label
                _set_speaker_meta(
                    segment,
                    source="text_cue",
                    confidence=0.99,
                    raw_label="cue_rule",
                    diarization_speaker=original_speaker,
                )
                continue

            if len(text) < self._min_text_characters:
                _set_speaker_meta(
                    segment,
                    source="diarization",
                    confidence=1.0,
                    raw_label=original_speaker,
                    diarization_speaker=original_speaker,
                    fallback_reason="short_text",
                )
                continue

            prediction = self._predict(text)
            if not prediction:
                _set_speaker_meta(
                    segment,
                    source="diarization",
                    confidence=1.0,
                    raw_label=original_speaker,
                    diarization_speaker=original_speaker,
                    fallback_reason="unmapped_model_label",
                )
                continue

            label, confidence, raw_label = prediction
            diarization_speaker = original_speaker
            votes = speaker_votes.setdefault(diarization_speaker, {"agent": 0.0, "customer": 0.0})
            votes[label] += confidence
            if confidence < self._min_confidence:
                _set_speaker_meta(
                    segment,
                    source="diarization",
                    confidence=confidence,
                    raw_label=raw_label,
                    diarization_speaker=original_speaker,
                    fallback_reason="low_confidence",
                )
                continue

            if label:
                segment["speaker"] = label
                _set_speaker_meta(
                    segment,
                    source="role_classifier",
                    confidence=confidence,
                    raw_label=raw_label,
                    diarization_speaker=original_speaker,
                )

        # Stabilize role assignment by diarization speaker identity when available.
        for segment in segments:
            meta = dict(segment.get("speaker_meta") or {})
            if meta.get("source") == "text_cue":
                continue
            diarization_speaker = str(meta.get("diarization_speaker") or "").strip()
            if not diarization_speaker:
                continue
            if diarization_speaker.lower() in {"unknown", "speaker", "spk"}:
                continue
            votes = speaker_votes.get(diarization_speaker)
            if not votes:
                continue
            winning_label = "agent" if votes["agent"] >= votes["customer"] else "customer"
            segment["speaker"] = winning_label
            meta["source"] = "cluster_vote"
            segment["speaker_meta"] = meta
        _smooth_unknown_speakers(segments)
        return segments

    def _predict(self, text: str) -> tuple[str, float, str] | None:
        if not self.is_available:
            return None

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        if "token_type_ids" in encoded:
            forward_sig = inspect.signature(self._model.forward)
            if "token_type_ids" not in forward_sig.parameters:
                encoded.pop("token_type_ids")
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            class_id = int(torch.argmax(logits, dim=-1).item())
            confidence = float(probs[0][class_id].item())

        id2label = getattr(self._model.config, "id2label", {}) or {}
        raw_label = str(id2label.get(class_id, "")).strip().lower()
        normalized = _normalize_speaker_label(raw_label)
        if not normalized:
            return None
        return normalized, confidence, raw_label


def _normalize_speaker_label(raw_label: str) -> str | None:
    label = raw_label.strip().lower()
    if not label:
        return None

    tokens = set(re.findall(r"[a-z0-9_]+", label))
    if tokens.intersection(_CUSTOMER_LABEL_ALIASES):
        return "customer"
    if tokens.intersection(_AGENT_LABEL_ALIASES):
        return "agent"
    return None


def _set_speaker_meta(
    segment: dict[str, Any],
    *,
    source: str,
    confidence: float | None = None,
    raw_label: str | None = None,
    fallback_reason: str | None = None,
    diarization_speaker: str | None = None,
) -> None:
    meta: dict[str, Any] = dict(segment.get("speaker_meta") or {})
    meta["source"] = source
    if confidence is not None:
        meta["confidence"] = float(confidence)
    if raw_label:
        meta["raw_label"] = raw_label
    if diarization_speaker:
        meta["diarization_speaker"] = diarization_speaker
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    segment["speaker_meta"] = meta


def _label_from_text_cues(text: str) -> str | None:
    normalized = (text or "").strip().lower()
    if not normalized:
        return None
    if any(cue in normalized for cue in _AGENT_TEXT_CUES):
        return "agent"
    if any(cue in normalized for cue in _CUSTOMER_TEXT_CUES):
        return "customer"
    return None


def _smooth_unknown_speakers(segments: list[dict[str, Any]]) -> None:
    def _norm(label: str | None) -> str:
        value = (label or "").strip().lower()
        return value if value in {"agent", "customer"} else "unknown"

    for idx, segment in enumerate(segments):
        if _norm(segment.get("speaker")) != "unknown":
            continue

        prev_label = "unknown"
        next_label = "unknown"
        for left in range(idx - 1, -1, -1):
            prev_label = _norm(segments[left].get("speaker"))
            if prev_label != "unknown":
                break
        for right in range(idx + 1, len(segments)):
            next_label = _norm(segments[right].get("speaker"))
            if next_label != "unknown":
                break

        chosen = None
        if prev_label == next_label and prev_label != "unknown":
            chosen = prev_label
        elif next_label != "unknown":
            chosen = next_label
        elif prev_label != "unknown":
            chosen = prev_label

        if chosen:
            segment["speaker"] = chosen
            meta = dict(segment.get("speaker_meta") or {})
            if "source" not in meta or meta.get("source") == "diarization":
                meta["source"] = "neighbor_smoothing"
            meta["fallback_reason"] = "unknown_smoothed"
            segment["speaker_meta"] = meta
