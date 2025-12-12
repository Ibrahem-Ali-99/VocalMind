"""Audio preprocessing utilities placeholder."""
from pathlib import Path
from typing import Any


class AudioProcessor:
    def load(self, path: Path) -> Any:
        """Load audio from disk; returns raw placeholder."""
        return path.read_bytes()

    def normalize(self, audio: Any) -> Any:
        """Normalize audio; stub for future implementation."""
        return audio

    def process(self, path: Path) -> Any:
        """Convenience wrapper to load and normalize audio."""
        audio = self.load(path)
        return self.normalize(audio)
