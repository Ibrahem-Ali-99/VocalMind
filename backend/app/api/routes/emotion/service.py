from app.core.config import settings
from app.core.inference_contracts import normalize_emotion_analysis
from app.core.kaggle_client import BaseKaggleClient


class EmotionAPIClient(BaseKaggleClient):
    local_endpoint = "/predict"
    remote_endpoint = "/emotion"

    @property
    def local_base_url(self) -> str:
        return settings.EMOTION_API_URL

    def normalize_response(self, data):
        return normalize_emotion_analysis(data)


emotion_client = EmotionAPIClient()
