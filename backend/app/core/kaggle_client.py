import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.core.inference_contracts import audio_content_type, is_supported_audio_filename


logger = logging.getLogger(__name__)

_LOCAL_SERVICE_FALLBACK_PORTS = {
    "whisperx": 8003,
    "emotion": 8001,
    "vad": 8002,
}


class BaseKaggleClient:
    local_endpoint: str = ""
    remote_endpoint: str = ""

    @property
    def local_base_url(self) -> str:
        raise NotImplementedError

    @property
    def remote_base_url(self) -> str:
        return settings.KAGGLE_SERVER_URL or settings.KAGGLE_NGROK_URL

    @property
    def endpoint(self) -> str:
        return self.local_endpoint if settings.IS_LOCAL else self.remote_endpoint

    @property
    def base_url(self) -> str:
        base_url = self.local_base_url if settings.IS_LOCAL else self.remote_base_url
        if not base_url:
            target = "local service" if settings.IS_LOCAL else "Kaggle server"
            raise HTTPException(status_code=500, detail=f"{target} URL is not configured.")
        return base_url

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.endpoint}"

    def headers(self) -> dict[str, str]:
        return {} if settings.IS_LOCAL else {"ngrok-skip-browser-warning": "true"}

    def normalize_response(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    async def analyze_audio(self, file: UploadFile) -> dict[str, Any]:
        content = await file.read()
        await file.seek(0)
        return await self.analyze_bytes(
            content,
            file.filename or "audio.wav",
            file.content_type or audio_content_type(file.filename),
        )

    async def analyze_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            filename,
            audio_bytes,
            content_type or audio_content_type(filename),
        )

    async def analyze_local_file(self, file_path: str) -> dict[str, Any]:
        if not is_supported_audio_filename(file_path):
            raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        with open(file_path, "rb") as file_handle:
            content = file_handle.read()

        return await self._post(
            os.path.basename(file_path),
            content,
            audio_content_type(file_path),
        )

    async def _post(self, filename: str, content: bytes, content_type: str) -> dict[str, Any]:
        urls_to_try = [self.url, *self._fallback_urls(self.url)]
        timeout = httpx.Timeout(300.0, connect=10.0)
        last_error: Exception | None = None
        response: httpx.Response | None = None

        for url in urls_to_try:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        url,
                        files={"file": (filename, content, content_type)},
                        headers=self.headers(),
                    )
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("%s timed out at %s: %s", self.endpoint, url, exc)
                continue
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning("%s unreachable at %s: %s", self.endpoint, url, exc)
                continue

            if response.status_code == 200:
                return self.normalize_response(response.json())

            logger.error("%s API error %s at %s: %s", self.endpoint, response.status_code, url, response.text)
            raise HTTPException(
                status_code=502,
                detail=f"{self.endpoint} service error: {response.text}",
            )

        if isinstance(last_error, httpx.TimeoutException):
            raise HTTPException(status_code=504, detail=f"{self.endpoint} service timed out.") from last_error

        target = "local service" if settings.IS_LOCAL else "Kaggle server"
        raise HTTPException(
            status_code=503,
            detail=f"{self.endpoint} service unreachable ({target}).",
        ) from last_error

    def _fallback_urls(self, primary_url: str) -> list[str]:
        if not settings.IS_LOCAL:
            return []

        parsed = urlparse(primary_url)
        host = (parsed.hostname or "").lower()
        fallback_port = _LOCAL_SERVICE_FALLBACK_PORTS.get(host)
        if fallback_port is None:
            return []

        path = parsed.path or self.endpoint
        fallback = urlunparse((parsed.scheme or "http", f"localhost:{fallback_port}", path, "", "", ""))
        if fallback == primary_url:
            return []
        return [fallback]
