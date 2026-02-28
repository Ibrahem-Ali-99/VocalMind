# Emotion microservice client.
# Logic: Proxies uploaded audio files to the GPU Docker container so the main backend
# stays free of heavy ML dependencies (PyTorch, CUDA, FunaSR).

import logging
from typing import Dict, Any

import httpx
from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmotionAPIClient:
    """HTTP client to the standalone emotion-api-gpu Docker container."""

    def __init__(self):
        self.predict_url = f"{settings.EMOTION_API_URL.rstrip('/')}/predict"

    async def analyze_audio(self, file: UploadFile) -> Dict[str, Any]:
        """Forward a .wav file to the GPU container and return the emotion result."""
        timeout = httpx.Timeout(60.0, connect=10.0)

        try:
            file_content = await file.read()
            await file.seek(0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.predict_url,
                    files={"file": (file.filename, file_content, file.content_type)},
                )

            if response.status_code == 200:
                return response.json()

            logger.error(f"Emotion API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail=f"Emotion service error: {response.text}")

        except httpx.RequestError as e:
            logger.error(f"Emotion API unreachable: {e}")
            raise HTTPException(
                status_code=503,
                detail="Emotion service unreachable. Ensure the emotion-api-gpu container is running.",
            )
        except httpx.TimeoutException:
            logger.error("Emotion API timed out.")
            raise HTTPException(status_code=504, detail="Emotion service timed out.")


# Module-level singleton
emotion_client = EmotionAPIClient()
