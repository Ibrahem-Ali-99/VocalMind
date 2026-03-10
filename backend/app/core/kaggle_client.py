# Shared HTTP client base for all Kaggle-proxied AI services.
# Subclasses set `endpoint` and `local_base_url` — all HTTP logic lives here.

import logging
import os
from typing import Dict, Any

import httpx
from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = (".wav", ".mp3")


class BaseKaggleClient:
    """Reusable async HTTP client for Kaggle / Docker AI endpoints."""

    endpoint: str = ""          # e.g. "/predict", "/diarize"
    local_base_url: str = ""    # Docker URL; subclass overrides via property or attr

    # ── URL resolution ──────────────────────────────────────────────

    @property
    def _base_url(self) -> str:
        if settings.IS_LOCAL:
            return self.local_base_url
        return settings.KAGGLE_NGROK_URL or settings.KAGGLE_SERVER_URL

    @property
    def url(self) -> str:
        return f"{self._base_url.rstrip('/')}{self.endpoint}"

    @staticmethod
    def _headers() -> Dict[str, str]:
        return {} if settings.IS_LOCAL else {"ngrok-skip-browser-warning": "true"}

    # ── Public API ──────────────────────────────────────────────────

    async def analyze_audio(self, file: UploadFile) -> Dict[str, Any]:
        """Forward a FastAPI UploadFile."""
        content = await file.read()
        await file.seek(0)
        return await self._post(file.filename, content, file.content_type or "audio/wav")

    async def analyze_bytes(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Forward raw bytes (used by pipeline callers)."""
        return await self._post(filename, audio_bytes, "audio/wav")

    async def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """Read a local file and forward it with tenacity retry."""
        if not file_path.endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        if not (settings.KAGGLE_NGROK_URL or settings.KAGGLE_SERVER_URL):
            raise HTTPException(status_code=500, detail="Kaggle URL is not configured.")

        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
            reraise=True,
        )
        async def _do():
            with open(file_path, "rb") as f:
                content = f.read()
            return await self._post(os.path.basename(file_path), content, "audio/wav")

        try:
            return await _do()
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"{self.endpoint} service unreachable after retries: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"{self.endpoint} service unreachable after retries. Check tunnel.",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error ({self.endpoint}): {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Core POST ───────────────────────────────────────────────────

    async def _post(self, filename: str, content: bytes, content_type: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(300.0, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.url,
                    files={"file": (filename, content, content_type)},
                    headers=self._headers(),
                )
            if response.status_code == 200:
                return response.json()
            logger.error(f"{self.endpoint} API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail=f"{self.endpoint} service error: {response.text}")
        except httpx.TimeoutException:
            logger.error(f"{self.endpoint} API timed out.")
            raise HTTPException(status_code=504, detail=f"{self.endpoint} service timed out.")
        except httpx.RequestError as e:
            logger.error(f"{self.endpoint} API unreachable: {e}")
            target = "Docker container" if settings.IS_LOCAL else "Kaggle server"
            raise HTTPException(status_code=503, detail=f"{self.endpoint} service unreachable ({target}).")
