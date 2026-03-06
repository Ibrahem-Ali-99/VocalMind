# Emotion microservice client.
# Logic: Proxies uploaded audio files to the GPU Docker container so the main backend
# stays free of heavy ML dependencies (PyTorch, CUDA, FunaSR).

import logging
import os
from typing import Dict, Any

import httpx
from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmotionAPIClient:
    """HTTP client to the standalone emotion-api-gpu Docker container."""

    def __init__(self):
        self.predict_url = f"{settings.EMOTION_API_URL.rstrip('/')}/predict"
        self.kaggle_url = f"{settings.KAGGLE_NGROK_URL.rstrip('/')}/analyze" if settings.KAGGLE_NGROK_URL else ""

    async def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """Forward a local file to the Kaggle ngrok URL with retries and authentication."""
        if not (file_path.endswith(".wav") or file_path.endswith(".mp3")):
            raise HTTPException(status_code=400, detail="Invalid file format for emotion analysis.")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        if not self.kaggle_url or "xxxx.ngrok-free.app" in self.kaggle_url:
             logger.error("KAGGLE_NGROK_URL is not configured properly.")
             raise HTTPException(status_code=500, detail="KAGGLE_NGROK_URL is not set.")

        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
            reraise=True
        )
        async def _do_post():
            timeout = httpx.Timeout(300.0, connect=10.0)
            headers = {"X-API-Key": settings.KAGGLE_API_SECRET}
            
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            filename = os.path.basename(file_path)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.kaggle_url,
                    files={"file": (filename, file_content, "audio/wav")},
                    headers=headers
                )

                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 403:
                     logger.error("Kaggle API Key mismatch!")
                     raise HTTPException(status_code=403, detail="Kaggle Worker authentication failed.")

                logger.error(f"Kaggle Emotion API error {response.status_code}: {response.text}")
                raise HTTPException(status_code=502, detail=f"Kaggle Emotion service error: {response.text}")

        try:
            return await _do_post()
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Kaggle Emotion API reached max retries or timed out: {e}")
            raise HTTPException(
                status_code=503,
                detail="Kaggle Emotion service unreachable after retries. Check ngrok tunnel.",
            )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Unexpected error in emotion analysis: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during analysis")

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

    async def analyze_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Forward raw audio bytes to the GPU container (used by the VAD pipeline)."""
        timeout = httpx.Timeout(60.0, connect=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.predict_url,
                    files={"file": (filename, audio_bytes, "audio/wav")},
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
