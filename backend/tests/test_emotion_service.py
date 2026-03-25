"""Tests for app.api.routes.emotion.service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, UploadFile

from app.api.routes.emotion.service import EmotionAPIClient


@pytest.fixture
def api():
    with patch("app.api.routes.emotion.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.EMOTION_API_URL = "http://local:8000"
        client_settings.IS_LOCAL = True
        client_settings.KAGGLE_SERVER_URL = "https://kaggle.example.com"
        client_settings.KAGGLE_NGROK_URL = ""
        yield EmotionAPIClient()


@pytest.mark.asyncio
async def test_analyze_bytes_normalizes_local_payload(api):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "emotion": "neutral",
        "confidence": 0.88,
        "raw_result": {"labels": ["neutral", "happy"], "scores": [0.88, 0.12]},
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
        result = await api.analyze_bytes(b"audio", "clip.wav")
    assert result["top_emotion"] == "neutral"
    assert result["top_score"] == 0.88
    assert result["emotions"][0]["label"] == "neutral"


@pytest.mark.asyncio
async def test_analyze_bytes_normalizes_remote_payload(api):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "top_emotion": "中立/neutral",
        "emotions": [{"label": "中立/neutral", "score": 0.91}],
    }
    with patch("app.core.kaggle_client.settings") as client_settings:
        client_settings.IS_LOCAL = False
        client_settings.KAGGLE_SERVER_URL = "https://kaggle.example.com"
        client_settings.KAGGLE_NGROK_URL = ""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_bytes(b"audio", "clip.wav")
    assert result["top_emotion"] == "neutral"
    assert result["emotions"][0]["label"] == "neutral"


def test_remote_url_and_headers_use_kaggle_endpoint():
    with patch("app.api.routes.emotion.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.EMOTION_API_URL = "http://local:8000"
        client_settings.IS_LOCAL = False
        client_settings.KAGGLE_SERVER_URL = "https://kaggle.example.com"
        client_settings.KAGGLE_NGROK_URL = ""
        api = EmotionAPIClient()
        assert api.url == "https://kaggle.example.com/emotion"
        assert api.headers() == {"ngrok-skip-browser-warning": "true"}


@pytest.mark.asyncio
async def test_analyze_bytes_502_on_error(api):
    resp = MagicMock(status_code=500, text="err")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
        with pytest.raises(HTTPException) as exc:
            await api.analyze_bytes(b"a", "c.wav")
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_analyze_bytes_503_on_connection_error(api):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.RequestError("refused")):
        with pytest.raises(HTTPException) as exc:
            await api.analyze_bytes(b"a", "c.wav")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_analyze_bytes_504_on_timeout(api):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("t")):
        with pytest.raises(HTTPException) as exc:
            await api.analyze_bytes(b"a", "c.wav")
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_analyze_audio_wrapper(api):
    import io

    file = UploadFile(filename="test.wav", file=io.BytesIO(b"audio"), headers={"content-type": "audio/wav"})
    with patch.object(api, "analyze_bytes", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = {"top_emotion": "happy"}
        result = await api.analyze_audio(file)
    assert result["top_emotion"] == "happy"
    mock_analyze.assert_called_once_with(b"audio", "test.wav", "audio/wav")


@pytest.mark.asyncio
async def test_analyze_local_file_invalid_ext(api):
    with pytest.raises(HTTPException) as exc:
        await api.analyze_local_file("recording.txt")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_analyze_local_file_not_found(api):
    with pytest.raises(HTTPException) as exc:
        await api.analyze_local_file("/nonexistent/path.wav")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_analyze_local_file_success(api):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "top_emotion": "开心/happy",
        "emotions": [{"label": "开心/happy", "score": 0.95}],
    }
    with patch("app.core.kaggle_client.os.path.exists", return_value=True), patch(
        "builtins.open", MagicMock()
    ), patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
        result = await api.analyze_local_file("test.wav")
    assert result["top_emotion"] == "happy"
