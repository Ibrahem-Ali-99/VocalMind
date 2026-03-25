from app.core.config import settings
from app.core.inference_contracts import normalize_transcription_response
from app.core.kaggle_client import BaseKaggleClient


class TranscriptionAPIClient(BaseKaggleClient):
    local_endpoint = "/transcribe"
    remote_endpoint = "/transcribe"

    @property
    def local_base_url(self) -> str:
        return settings.WHISPERX_API_URL

    def normalize_response(self, data):
        return normalize_transcription_response(data)


transcription_client = TranscriptionAPIClient()
