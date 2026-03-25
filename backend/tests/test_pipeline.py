"""Tests for app.api.routes.emotion.pipeline."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.api.routes.emotion.pipeline import _local_vad_url, process_audio


INTERACTION_ID = uuid4()


def _segment(index=0, start=0.0, end=1.0):
    return {
        "index": index,
        "start_time": start,
        "end_time": end,
        "audio_base64": base64.b64encode(b"clip").decode(),
    }


def _vad_response(segments):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"segments": segments}
    return resp


def test_local_vad_url():
    with patch("app.api.routes.emotion.pipeline.settings") as s:
        s.VAD_API_URL = "http://vad:8002"
        assert _local_vad_url() == "http://vad:8002/split"


@pytest.mark.asyncio
async def test_process_audio_local_success():
    emotion = {"top_emotion": "angry", "top_score": 0.91}
    with patch("app.api.routes.emotion.pipeline.settings") as s, patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=_vad_response([_segment(0, 0.0, 1.5)]),
    ), patch(
        "app.api.routes.emotion.pipeline.emotion_client.analyze_bytes",
        new_callable=AsyncMock,
        return_value=emotion,
    ):
        s.IS_LOCAL = True
        s.VAD_API_URL = "http://vad:8002"
        result = await process_audio(b"audio", "call.wav", INTERACTION_ID)
    assert len(result) == 1
    assert result[0]["emotion"] == "angry"
    assert result[0]["interaction_id"] == str(INTERACTION_ID)


@pytest.mark.asyncio
async def test_process_audio_remote_success():
    full_result = {
        "top_emotion": "neutral",
        "top_score": 0.93,
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "Hello there",
                "speaker": "UNKNOWN",
                "emotion": "neutral",
                "emotion_scores": [{"label": "neutral", "score": 0.93}],
            }
        ],
    }
    with patch("app.api.routes.emotion.pipeline.settings") as s, patch(
        "app.api.routes.emotion.pipeline.full_client.analyze_bytes",
        new_callable=AsyncMock,
        return_value=full_result,
    ):
        s.IS_LOCAL = False
        result = await process_audio(b"audio", "call.wav", INTERACTION_ID)
    assert len(result) == 1
    assert result[0]["emotion"] == "neutral"
    assert result[0]["text"] == "Hello there"
    assert result[0]["speaker_role"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_process_audio_local_no_segments():
    with patch("app.api.routes.emotion.pipeline.settings") as s, patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=_vad_response([]),
    ):
        s.IS_LOCAL = True
        s.VAD_API_URL = "http://vad:8002"
        assert await process_audio(b"audio", "call.wav", INTERACTION_ID) == []


@pytest.mark.asyncio
async def test_process_audio_local_vad_unreachable():
    with patch("app.api.routes.emotion.pipeline.settings") as s, patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=httpx.RequestError("refused"),
    ):
        s.IS_LOCAL = True
        s.VAD_API_URL = "http://vad:8002"
        with pytest.raises(HTTPException) as exc:
            await process_audio(b"a", "c.wav", INTERACTION_ID)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_process_audio_local_vad_error():
    resp = MagicMock(status_code=500, text="GPU OOM")
    with patch("app.api.routes.emotion.pipeline.settings") as s, patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=resp,
    ):
        s.IS_LOCAL = True
        s.VAD_API_URL = "http://vad:8002"
        with pytest.raises(HTTPException) as exc:
            await process_audio(b"a", "c.wav", INTERACTION_ID)
    assert exc.value.status_code == 502
