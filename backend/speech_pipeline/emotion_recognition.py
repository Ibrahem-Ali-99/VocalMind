"""Emotion recognition placeholder."""
from typing import Dict


class EmotionRecognizer:
    def predict(self, audio_bytes: bytes) -> Dict[str, float]:
        """Return dummy emotion probabilities."""
        return {"neutral": 0.7, "happy": 0.2, "stressed": 0.1}
