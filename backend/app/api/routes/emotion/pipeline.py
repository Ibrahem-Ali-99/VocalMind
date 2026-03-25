import base64
import logging
from typing import Any
from uuid import UUID

import httpx
from fastapi import HTTPException

from app.api.routes.emotion.service import emotion_client
from app.api.routes.full.service import full_client
from app.core.config import settings
from app.core.inference_contracts import normalize_emotion_label


logger = logging.getLogger(__name__)


def _local_vad_url() -> str:
    return f"{settings.VAD_API_URL.rstrip('/')}/split"


async def _process_local_audio(
    audio_bytes: bytes,
    filename: str,
    interaction_id: UUID,
) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _local_vad_url(),
                files={"file": (filename, audio_bytes, "audio/wav")},
            )
    except httpx.RequestError as exc:
        logger.error("VAD service unreachable (local service): %s", exc)
        raise HTTPException(status_code=503, detail="VAD service unreachable (local service).") from exc

    if response.status_code != 200:
        logger.error("VAD service error %s: %s", response.status_code, response.text)
        raise HTTPException(status_code=502, detail=f"VAD service error: {response.text}")

    segments = response.json().get("segments", [])
    if not segments:
        logger.warning("VAD found no speech segments.")
        return []

    utterances: list[dict[str, Any]] = []
    for seg in segments:
        clip_bytes = base64.b64decode(seg["audio_base64"])
        emotion_result = await emotion_client.analyze_bytes(
            clip_bytes,
            f"segment_{seg['index']}.wav",
        )
        utterances.append(
            {
                "interaction_id": str(interaction_id),
                "transcript_id": None,
                "speaker_role": None,
                "user_id": None,
                "sequence_index": seg["index"],
                "start_time_seconds": seg["start_time"],
                "end_time_seconds": seg["end_time"],
                "text": None,
                "emotion": emotion_result.get("top_emotion"),
                "emotion_confidence": emotion_result.get("top_score"),
            }
        )
    return utterances


async def _process_remote_audio(
    audio_bytes: bytes,
    filename: str,
    interaction_id: UUID,
) -> list[dict[str, Any]]:
    full_result = await full_client.analyze_bytes(audio_bytes, filename)
    utterances: list[dict[str, Any]] = []

    for index, seg in enumerate(full_result.get("segments", [])):
        emotion_scores = seg.get("emotion_scores") or []
        confidence = emotion_scores[0]["score"] if emotion_scores else full_result.get("top_score")
        utterances.append(
            {
                "interaction_id": str(interaction_id),
                "transcript_id": None,
                "speaker_role": seg.get("speaker"),
                "user_id": None,
                "sequence_index": index,
                "start_time_seconds": seg.get("start"),
                "end_time_seconds": seg.get("end"),
                "text": seg.get("text"),
                "emotion": normalize_emotion_label(seg.get("emotion")),
                "emotion_confidence": confidence,
            }
        )

    return utterances


async def process_audio(
    audio_bytes: bytes,
    filename: str,
    interaction_id: UUID,
) -> list[dict[str, Any]]:
    if settings.IS_LOCAL:
        return await _process_local_audio(audio_bytes, filename, interaction_id)
    return await _process_remote_audio(audio_bytes, filename, interaction_id)
