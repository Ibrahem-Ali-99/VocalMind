"""Sentiment analysis placeholder."""
from typing import Dict


class SentimentAnalyzer:
    def analyze(self, text: str) -> Dict[str, float]:
        """Return dummy sentiment scores."""
        return {"positive": 0.5, "neutral": 0.3, "negative": 0.2}
