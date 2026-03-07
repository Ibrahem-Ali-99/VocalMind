"""Tests for app.api.routes.emotion.pipeline — audio processing orchestration."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.api.routes.emotion.pipeline import _vad_url, process_audio

INTERACTION_ID = uuid4()


def _segment(index=0, start=0.0, end=1.0):
    return {
        "index": index, "start_time": start, "end_time": end,
        "audio_base64": base64.b64encode(b"clip").decode(),
    }


def _vad_response(segments):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"segments": segments}
    return resp


def test_vad_url_local():
    with patch("app.api.routes.emotion.pipeline.settings") as s:
        s.IS_LOCAL, s.VAD_API_URL = True, "http://vad:8002"
        assert _vad_url() == "http://vad:8002/split"


def test_vad_url_kaggle():
    with patch("app.api.routes.emotion.pipeline.settings") as s:
        s.IS_LOCAL, s.KAGGLE_SERVER_URL = False, "https://kaggle.example.com/"
        assert _vad_url() == "https://kaggle.example.com/split"


@pytest.mark.asyncio
async def test_process_audio_success():
    emotion = {"emotion": "angry", "confidence": 0.91}
    with (
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_vad_response([_segment(0, 0.0, 1.5)])),
        patch("app.api.routes.emotion.pipeline.emotion_client.analyze_bytes", new_callable=AsyncMock, return_value=emotion),
    ):
        result = await process_audio(b"audio", "call.wav", INTERACTION_ID)
    assert len(result) == 1
    assert result[0]["emotion"] == "angry"
    assert result[0]["interaction_id"] == str(INTERACTION_ID)


@pytest.mark.asyncio
async def test_process_audio_no_segments():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_vad_response([])):
        assert await process_audio(b"audio", "call.wav", INTERACTION_ID) == []


@pytest.mark.asyncio
async def test_process_audio_vad_unreachable():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.RequestError("refused")):
        with pytest.raises(HTTPException) as exc:
            await process_audio(b"a", "c.wav", INTERACTION_ID)
        assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_process_audio_vad_error():
    resp = MagicMock(status_code=500, text="GPU OOM")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
        with pytest.raises(HTTPException) as exc:
            await process_audio(b"a", "c.wav", INTERACTION_ID)
        assert exc.value.status_code == 502
