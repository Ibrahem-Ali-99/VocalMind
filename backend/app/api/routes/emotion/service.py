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

    def __init__(self):
        # We can still use the property for dynamic URL resolution
        pass

    @property
    def predict_url(self) -> str:
        if settings.IS_LOCAL:
            base = settings.EMOTION_API_URL
        else:
            base = settings.KAGGLE_NGROK_URL or settings.KAGGLE_SERVER_URL
        
        # Most endpoints use /predict, but the local Kaggle worker used /analyze
        # For backward compatibility with the existing analyze_local_file logic:
        return f"{base.rstrip('/')}/predict"

    @property
    def kaggle_analyze_url(self) -> str:
        base = settings.KAGGLE_NGROK_URL or settings.KAGGLE_SERVER_URL
        return f"{base.rstrip('/')}/analyze"

    async def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """Forward a local file to the Kaggle ngrok URL with retries and authentication."""
        if not (file_path.endswith(".wav") or file_path.endswith(".mp3")):
            raise HTTPException(status_code=400, detail="Invalid file format for emotion analysis.")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        if not (settings.KAGGLE_NGROK_URL or settings.KAGGLE_SERVER_URL):
             logger.error("Kaggle URL is not configured properly.")
             raise HTTPException(status_code=500, detail="Kaggle URL is not set.")

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
            if not settings.IS_LOCAL:
                headers["ngrok-skip-browser-warning"] = "true"
            
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            filename = os.path.basename(file_path)

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Use kaggle_analyze_url specifically for this legacy/robust method
                response = await client.post(
                    self.kaggle_analyze_url,
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
        """Forward an UploadFile to the emotion service."""
        file_content = await file.read()
        await file.seek(0)
        return await self._post(file.filename, file_content, file.content_type or "audio/wav")

    async def analyze_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Forward raw audio bytes to the emotion service (used by the VAD pipeline)."""
        return await self._post(filename, audio_bytes, "audio/wav")

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
