"""Tests for app.api.routes.emotion.service — EmotionAPIClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.api.routes.emotion.service import EmotionAPIClient


@pytest.fixture
def api():
    with patch("app.api.routes.emotion.service.settings") as mock_settings:
        mock_settings.IS_LOCAL = True
        mock_settings.EMOTION_API_URL = "http://local:8000"
        mock_settings.KAGGLE_NGROK_URL = "http://kaggle:8000"
        mock_settings.KAGGLE_API_SECRET = "secret"
        yield EmotionAPIClient()


@pytest.mark.asyncio
async def test_analyze_bytes_success(api):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"emotion": "neutral", "confidence": 0.88}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
        result = await api.analyze_bytes(b"audio", "clip.wav")
    assert result["emotion"] == "neutral"


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
    from fastapi import UploadFile
    import io
    file = UploadFile(filename="test.wav", file=io.BytesIO(b"audio"), headers={"content-type": "audio/wav"})
    with patch.object(api, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"emotion": "happy"}
        result = await api.analyze_audio(file)
        assert result["emotion"] == "happy"
        mock_post.assert_called_once_with("test.wav", b"audio", "audio/wav")


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
    # Mock settings to use Kaggle URL
    with patch("app.api.routes.emotion.service.settings") as mock_settings, \
         patch("app.api.routes.emotion.service.os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()):
        mock_settings.IS_LOCAL = False
        mock_settings.KAGGLE_NGROK_URL = "http://kaggle:8000"
        mock_settings.KAGGLE_API_SECRET = "secret"
        
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"emotion": "happy"}
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_local_file("test.wav")
            assert result["emotion"] == "happy"


@pytest.mark.asyncio
async def test_analyze_local_file_403_auth_error(api):
    with patch("app.api.routes.emotion.service.settings") as mock_settings, \
         patch("app.api.routes.emotion.service.os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()):
        mock_settings.IS_LOCAL = False
        mock_settings.KAGGLE_NGROK_URL = "http://kaggle:8000"
        
        resp = MagicMock(status_code=403, text="forbidden")
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(HTTPException) as exc:
                await api.analyze_local_file("test.wav")
            assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_analyze_local_file_503_on_max_retries(api):
    with patch("app.api.routes.emotion.service.settings") as mock_settings, \
         patch("app.api.routes.emotion.service.os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()):
        mock_settings.IS_LOCAL = False
        mock_settings.KAGGLE_NGROK_URL = "http://kaggle:8000"
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("failed")):
            # We need to speed up the retry wait for tests
            with patch("tenacity.nap.time.sleep", return_value=None):
                 with pytest.raises(HTTPException) as exc:
                     await api.analyze_local_file("test.wav")
                 assert exc.value.status_code == 503
