"""Optional DistilBERT speaker-role relabeling for transcript segments (backend pipeline).

Set ``SPEAKER_ROLE_MODEL_DIR`` to a Hugging Face model directory (same export as WhisperX uses).
If unset, import fails, or inference errors, segments pass through unchanged.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_classifier: Any = None
_CUSTOMER_LABEL_ALIASES = {"customer", "client", "caller", "user", "speaker_0", "speaker0", "s0"}
_AGENT_LABEL_ALIASES = {"agent", "support", "representative", "advisor", "speaker_1", "speaker1", "s1"}


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


class _SpeakerRoleClassifier:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._failed = False

    @property
    def is_available(self) -> bool:
        return not self._failed and self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if self._failed or self.is_available:
            return
        if not self._model_dir.exists():
            logger.info("Speaker role model directory missing at %s; skipping relabel.", self._model_dir)
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self._model_dir)).to("cpu").eval()
            logger.info("Loaded speaker-role classifier from %s", self._model_dir)
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._failed = True
            self._model = None
            self._tokenizer = None
            logger.warning("Speaker-role classifier unavailable: %s", exc)

    def _predict(self, text: str) -> str | None:
        if not self.is_available:
            return None
        import torch

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
            class_id = int(torch.argmax(logits, dim=-1).item())

        id2label = getattr(self._model.config, "id2label", {}) or {}
        raw_label = str(id2label.get(class_id, "")).strip().lower()
        return _normalize_speaker_label(raw_label)

    def relabel_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return segments
        self.load()
        if not self.is_available:
            return segments
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            label = self._predict(text)
            if label:
                segment["speaker"] = label
        return segments


def relabel_segments_with_speaker_model(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply optional binary speaker-role classifier in-process."""
    global _classifier
    if not settings.BACKEND_SPEAKER_RELABEL_ENABLED:
        return segments
    raw = (settings.SPEAKER_ROLE_MODEL_DIR or "").strip()
    if not raw:
        return segments
    model_dir = Path(raw)
    if _classifier is None:
        _classifier = _SpeakerRoleClassifier(model_dir)
    return _classifier.relabel_segments(segments)
