from app.core.config import settings
from app.core.inference_contracts import normalize_vad_response
from app.core.kaggle_client import BaseKaggleClient


class VADAPIClient(BaseKaggleClient):
    local_endpoint = "/split"
    remote_endpoint = "/vad"

    @property
    def local_base_url(self) -> str:
        return settings.VAD_API_URL

    def normalize_response(self, data):
        return normalize_vad_response(data)


vad_client = VADAPIClient()
