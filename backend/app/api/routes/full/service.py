from app.api.routes.emotion.service import emotion_client
from app.api.routes.transcription.service import transcription_client
from app.core.config import settings
from app.core.inference_contracts import build_local_full_response, normalize_full_response
from app.core.kaggle_client import BaseKaggleClient


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
        emotion_analysis = await emotion_client.analyze_bytes(audio_bytes, filename, content_type)
        return build_local_full_response(transcription, emotion_analysis)

    async def analyze_local_file(self, file_path: str):
        if not settings.IS_LOCAL:
            return await super().analyze_local_file(file_path)

        transcription = await transcription_client.analyze_local_file(file_path)
        emotion_analysis = await emotion_client.analyze_local_file(file_path)
        return build_local_full_response(transcription, emotion_analysis)


full_client = FullAPIClient()
