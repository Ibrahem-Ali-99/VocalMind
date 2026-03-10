# Diarization microservice client.
# Inherits all HTTP logic from BaseKaggleClient — only sets the endpoint.

from app.core.config import settings
from app.core.kaggle_client import BaseKaggleClient


class DiarizationAPIClient(BaseKaggleClient):
    """HTTP client for the speaker diarization service."""

    endpoint = "/diarize"

    @property
    def local_base_url(self) -> str:
        return settings.EMOTION_API_URL  # same Docker host for now


# Module-level singleton
diarization_client = DiarizationAPIClient()
