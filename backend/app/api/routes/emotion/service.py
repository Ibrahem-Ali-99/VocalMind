# Emotion microservice client.
# Inherits shared HTTP logic from BaseKaggleClient.
# Only overrides analyze_local_file for the legacy /analyze endpoint + API-key auth.

import logging
import os
from typing import Dict, Any

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.kaggle_client import BaseKaggleClient, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


class EmotionAPIClient(BaseKaggleClient):
    """HTTP client for the emotion recognition service."""

    endpoint = "/predict"

    @property
    def local_base_url(self) -> str:
        return settings.EMOTION_API_URL

    @property
    def _kaggle_analyze_url(self) -> str:
        """Legacy /analyze URL used only by analyze_local_file."""
        base = settings.KAGGLE_NGROK_URL or settings.KAGGLE_SERVER_URL
        return f"{base.rstrip('/')}/analyze"

    async def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """Override: uses /analyze endpoint with X-API-Key header and tenacity retry."""
        if not file_path.endswith(ALLOWED_EXTENSIONS):
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
            reraise=True,
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
                response = await client.post(
                    self._kaggle_analyze_url,
                    files={"file": (filename, file_content, "audio/wav")},
                    headers=headers,
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
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in emotion analysis: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during analysis")


# Module-level singleton
emotion_client = EmotionAPIClient()
