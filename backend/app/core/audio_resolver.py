from __future__ import annotations

from pathlib import Path

import httpx

from app.core.config import settings

_AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
}


def audio_media_type_from_path(audio_path: str) -> str:
    ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
    return _AUDIO_CONTENT_TYPES.get(ext, "audio/wav")


def audio_filename_from_path(audio_path: str) -> str:
    name = Path(audio_path).name.strip()
    return name or "audio.wav"


def resolve_local_audio_path(audio_path: str) -> Path | None:
    backend_dir = Path(__file__).resolve().parents[2]
    candidates = [Path(audio_path), backend_dir / audio_path]
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _storage_url_for_path(audio_path: str) -> str:
    return f"{settings.SUPABASE_URL}/storage/v1/object/{audio_path}"


async def fetch_supabase_audio(audio_path: str, timeout_seconds: float = 60.0) -> bytes:
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        raise FileNotFoundError("Supabase storage is not configured")

    storage_url = _storage_url_for_path(audio_path)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            storage_url,
            headers={
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
                "apikey": settings.SUPABASE_SERVICE_KEY,
            },
            timeout=timeout_seconds,
        )

    if response.status_code != 200:
        raise FileNotFoundError(f"Supabase object not found: {audio_path}")

    return response.content


async def supabase_object_exists(audio_path: str, timeout_seconds: float = 10.0) -> bool:
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        return False

    storage_url = _storage_url_for_path(audio_path)
    headers = {
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
        "apikey": settings.SUPABASE_SERVICE_KEY,
    }
    async with httpx.AsyncClient() as client:
        response = await client.head(storage_url, headers=headers, timeout=timeout_seconds)
        if response.status_code == 200:
            return True
        if response.status_code not in {404, 405}:
            return False
        # Some storage gateways disable HEAD. Fall back to a 1-byte range GET.
        response = await client.get(
            storage_url,
            headers={**headers, "Range": "bytes=0-0"},
            timeout=timeout_seconds,
        )
    return response.status_code in {200, 206}


async def fetch_audio_bytes(audio_path: str, timeout_seconds: float = 60.0) -> tuple[bytes, str]:
    local_path = resolve_local_audio_path(audio_path)
    if local_path:
        return local_path.read_bytes(), local_path.name

    return await fetch_supabase_audio(audio_path, timeout_seconds=timeout_seconds), audio_filename_from_path(audio_path)
