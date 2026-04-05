import logging
import asyncio

from app.api.routes.emotion.service import emotion_client
from app.api.routes.transcription.service import transcription_client
from app.core.config import settings
from app.core.emotion_fusion import build_deterministic_emotion_analysis
from app.core.inference_contracts import build_local_full_response, normalize_full_response
from app.core.kaggle_client import BaseKaggleClient


logger = logging.getLogger(__name__)


def _deterministic_emotion_fallback(transcript_text: str) -> dict:
    return build_deterministic_emotion_analysis(transcript_text)


class FullAPIClient(BaseKaggleClient):
    local_endpoint = "/full"
    remote_endpoint = "/full"

    @property
    def local_base_url(self) -> str:
        return settings.WHISPERX_API_URL

    def normalize_response(self, data):
        return normalize_full_response(data)

    async def analyze_bytes(self, audio_bytes, filename, content_type=None):
        if not settings.IS_LOCAL:
            return await super().analyze_bytes(audio_bytes, filename, content_type)

        transcription = await transcription_client.analyze_bytes(audio_bytes, filename, content_type)
        transcription_text = (transcription.get("text") or "").strip()
        emotion_analysis = _deterministic_emotion_fallback(transcription_text)

        for attempt in range(1, 4):
            try:
                emotion_analysis = await emotion_client.analyze_bytes(audio_bytes, filename, content_type)
                break
            except Exception as exc:
                if attempt == 3:
                    logger.warning("Emotion inference failed for %s after retries, using fallback: %s", filename, exc)
                else:
                    await asyncio.sleep(0.4 * attempt)

        if emotion_analysis.get("top_emotion") in {"neutral", "unknown"}:
            deterministic_emotion = _deterministic_emotion_fallback(transcription_text)
            if deterministic_emotion.get("top_emotion") not in {"neutral", "unknown"}:
                emotion_analysis = deterministic_emotion

        return build_local_full_response(transcription, emotion_analysis)

    async def analyze_local_file(self, file_path: str):
        if not settings.IS_LOCAL:
            return await super().analyze_local_file(file_path)

        transcription = await transcription_client.analyze_local_file(file_path)
        transcription_text = (transcription.get("text") or "").strip()
        emotion_analysis = _deterministic_emotion_fallback(transcription_text)

        for attempt in range(1, 4):
            try:
                emotion_analysis = await emotion_client.analyze_local_file(file_path)
                break
            except Exception as exc:
                if attempt == 3:
                    logger.warning("Emotion inference failed for %s after retries, using fallback: %s", file_path, exc)
                else:
                    await asyncio.sleep(0.4 * attempt)

        if emotion_analysis.get("top_emotion") in {"neutral", "unknown"}:
            deterministic_emotion = _deterministic_emotion_fallback(transcription_text)
            if deterministic_emotion.get("top_emotion") not in {"neutral", "unknown"}:
                emotion_analysis = deterministic_emotion
        return build_local_full_response(transcription, emotion_analysis)


full_client = FullAPIClient()
