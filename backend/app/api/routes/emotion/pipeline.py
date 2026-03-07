# Audio Processing Pipeline — Orchestrator
#
# The backend is a thin orchestrator that calls two microservices:
#   1. VAD  (POST /split)   → returns timestamped audio clips
#   2. Emotion (POST /predict) → returns emotion per clip
#
# Routes to Docker or Kaggle based on IS_LOCAL setting.

import base64
import logging
from typing import List, Dict, Any
from uuid import UUID

import httpx
from fastapi import HTTPException

from app.api.routes.emotion.service import emotion_client
from app.core.config import settings

logger = logging.getLogger(__name__)


def _vad_url() -> str:
    base = settings.VAD_API_URL if settings.IS_LOCAL else settings.KAGGLE_SERVER_URL
    return f"{base.rstrip('/')}/split"


async def process_audio(
    audio_bytes: bytes,
    filename: str,
    interaction_id: UUID,
) -> List[Dict[str, Any]]:
    """
    Orchestration pipeline:
      1. Send full audio to VAD service → get segments + base64 clips
      2. Send each clip to Emotion service → get emotion label
      3. Aggregate into utterance-shaped dicts
    """

    # ── Step 1: Call VAD Service ────────────────────────────────────
    timeout = httpx.Timeout(120.0, connect=10.0)
    headers = {} if settings.IS_LOCAL else {"ngrok-skip-browser-warning": "true"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _vad_url(),
                files={"file": (filename, audio_bytes, "audio/wav")},
                headers=headers,
            )
    except httpx.RequestError as e:
        target = "Docker container" if settings.IS_LOCAL else "Kaggle server"
        logger.error(f"VAD service unreachable ({target}): {e}")
        raise HTTPException(status_code=503, detail=f"VAD service unreachable ({target}).")

    if response.status_code != 200:
        logger.error(f"VAD service error {response.status_code}: {response.text}")
        raise HTTPException(status_code=502, detail=f"VAD service error: {response.text}")

    vad_result = response.json()
    segments = vad_result.get("segments", [])

    if not segments:
        logger.warning("VAD found no speech segments.")
        return []

    logger.info(f"VAD returned {len(segments)} segment(s).")

    # ── Step 2: Emotion per segment ─────────────────────────────────
    utterances: List[Dict[str, Any]] = []

    for seg in segments:
        clip_bytes = base64.b64decode(seg["audio_base64"])

        emotion_result = await emotion_client.analyze_bytes(
            clip_bytes, f"segment_{seg['index']}.wav"
        )

        utterances.append({
            "interaction_id": str(interaction_id),
            "transcript_id": None,
            "speaker_role": None,
            "user_id": None,
            "sequence_index": seg["index"],
            "start_time_seconds": seg["start_time"],
            "end_time_seconds": seg["end_time"],
            "text": None,
            "emotion": emotion_result.get("emotion"),
            "emotion_confidence": emotion_result.get("confidence"),
        })

    return utterances
