from app.core.config import settings
from app.core.inference_contracts import normalize_diarization_response
from app.core.kaggle_client import BaseKaggleClient


class DiarizationAPIClient(BaseKaggleClient):
    local_endpoint = "/transcribe"
    remote_endpoint = "/diarize"

    @property
    def local_base_url(self) -> str:
        return settings.WHISPERX_API_URL

    def normalize_response(self, data):
        return normalize_diarization_response(data)


diarization_client = DiarizationAPIClient()
