# Emotion microservice client.
# Routes requests to Docker container (IS_LOCAL=true) or Kaggle server (IS_LOCAL=false).

import logging
import os
from typing import Dict, Any

import httpx
from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmotionAPIClient:
    """HTTP client for the emotion recognition service."""

    @property
    def predict_url(self) -> str:
        base = settings.EMOTION_API_URL if settings.IS_LOCAL else settings.KAGGLE_SERVER_URL
        return f"{base.rstrip('/')}/predict"

    async def analyze_audio(self, file: UploadFile) -> Dict[str, Any]:
        """Forward an UploadFile to the emotion service."""
        file_content = await file.read()
        await file.seek(0)
        return await self._post(file.filename, file_content, file.content_type or "audio/wav")

    async def analyze_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Forward raw audio bytes to the emotion service (used by the VAD pipeline)."""
        return await self._post(filename, audio_bytes, "audio/wav")

    async def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """Forward a local file to the emotion service."""
        if not (file_path.endswith(".wav") or file_path.endswith(".mp3")):
            raise HTTPException(status_code=400, detail="Invalid file format for emotion analysis.")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_content = f.read()
        return await self._post(os.path.basename(file_path), file_content, "audio/wav")

    async def _post(self, filename: str, content: bytes, content_type: str) -> Dict[str, Any]:
        """Shared POST logic with timeout and error handling."""
        timeout = httpx.Timeout(300.0, connect=10.0)
        headers = {} if settings.IS_LOCAL else {"ngrok-skip-browser-warning": "true"}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.predict_url,
                    files={"file": (filename, content, content_type)},
                    headers=headers,
                )
            if response.status_code == 200:
                return response.json()
            logger.error(f"Emotion API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail=f"Emotion service error: {response.text}")
        except httpx.TimeoutException:
            logger.error("Emotion API timed out.")
            raise HTTPException(status_code=504, detail="Emotion service timed out.")
        except httpx.RequestError as e:
            logger.error(f"Emotion API unreachable: {e}")
            target = "Docker container" if settings.IS_LOCAL else "Kaggle server"
            raise HTTPException(status_code=503, detail=f"Emotion service unreachable ({target}).")


# Module-level singleton
emotion_client = EmotionAPIClient()
