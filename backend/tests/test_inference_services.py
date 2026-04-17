from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.routes.diarization.service import DiarizationAPIClient
from app.api.routes.full.service import FullAPIClient
from app.api.routes.transcription.service import TranscriptionAPIClient
from app.api.routes.vad.service import VADAPIClient


@pytest.mark.asyncio
async def test_transcription_service_normalizes_segments_without_top_level_text():
    with patch("app.api.routes.transcription.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = TranscriptionAPIClient()

        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["text"] == "hello world"
    assert result["segments"][1]["text"] == "world"


@pytest.mark.asyncio
async def test_transcription_service_preserves_speaker_metadata():
    with patch("app.api.routes.transcription.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = TranscriptionAPIClient()

        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": "SPEAKER_00",
                    "speaker_meta": {"source": "diarization", "confidence": 1.0},
                }
            ],
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["segments"][0]["speaker_meta"]["source"] == "diarization"


@pytest.mark.asyncio
async def test_vad_service_normalizes_local_split_payload():
    with patch("app.api.routes.vad.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.VAD_API_URL = "http://vad:8002"
        client_settings.IS_LOCAL = True
        api = VADAPIClient()

        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "segments": [{"index": 0, "start_time": 0.0, "end_time": 1.25, "audio_base64": "abc"}]
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result == {"speech_segments": [{"start": 0.0, "end": 1.25}]}


@pytest.mark.asyncio
async def test_diarization_service_normalizes_local_whisperx_payload():
    with patch("app.api.routes.diarization.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = DiarizationAPIClient()

        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "world", "speaker": "SPEAKER_01"},
            ]
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][1]["end"] == 2.0


@pytest.mark.asyncio
async def test_full_service_builds_local_response_from_transcription_and_emotion():
    with patch("app.api.routes.full.service.settings") as service_settings, patch(
        "app.api.routes.emotion.service.settings"
    ) as emotion_settings, patch("app.api.routes.transcription.service.settings") as transcription_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        emotion_settings.EMOTION_API_URL = "http://emotion:8000"
        transcription_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = FullAPIClient()

        with patch(
            "app.api.routes.full.service.transcription_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "text": "hello world",
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
            },
        ), patch(
            "app.api.routes.full.service.emotion_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "top_emotion": "neutral",
                "top_score": 0.9,
                "emotions": [{"label": "neutral", "score": 0.9}],
            },
        ):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["top_emotion"] == "neutral"
    assert result["segments"][0]["emotion"] == "neutral"
    assert result["segments"][0]["speaker"] == "SPEAKER_00"


@pytest.mark.asyncio
async def test_full_service_derives_segment_level_emotions_from_segment_text():
    with patch("app.api.routes.full.service.settings") as service_settings, patch(
        "app.api.routes.emotion.service.settings"
    ) as emotion_settings, patch("app.api.routes.transcription.service.settings") as transcription_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        emotion_settings.EMOTION_API_URL = "http://emotion:8000"
        transcription_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = FullAPIClient()

        with patch(
            "app.api.routes.full.service.transcription_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "text": "This is terrible and unacceptable. Thank you for helping me today.",
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "This is terrible and unacceptable.",
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Thank you for helping me today.",
                        "speaker": "SPEAKER_01",
                    },
                ],
            },
        ), patch(
            "app.api.routes.full.service.emotion_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "top_emotion": "happy",
                "top_score": 0.97,
                "emotions": [{"label": "happy", "score": 0.97}],
            },
        ):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["top_emotion"] == "happy"
    assert result["segments"][0]["emotion"] == "angry"
    assert result["segments"][1]["emotion"] == "happy"
    assert result["segments"][0]["emotion_scores"][0]["label"] == "angry"


@pytest.mark.asyncio
async def test_full_service_uses_deterministic_text_emotion_when_service_returns_neutral():
    with patch("app.api.routes.full.service.settings") as service_settings, patch(
        "app.api.routes.emotion.service.settings"
    ) as emotion_settings, patch("app.api.routes.transcription.service.settings") as transcription_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        emotion_settings.EMOTION_API_URL = "http://emotion:8000"
        transcription_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = FullAPIClient()

        with patch(
            "app.api.routes.full.service.transcription_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "text": "This is terrible and unacceptable.",
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "This is terrible and unacceptable.", "speaker": "SPEAKER_00"}],
            },
        ), patch(
            "app.api.routes.full.service.emotion_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "top_emotion": "neutral",
                "top_score": 0.9,
                "emotions": [{"label": "neutral", "score": 0.9}],
            },
        ):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["top_emotion"] == "angry"
    assert result["segments"][0]["emotion"] == "angry"


@pytest.mark.asyncio
async def test_full_service_uses_deterministic_text_emotion_for_complaints():
    with patch("app.api.routes.full.service.settings") as service_settings, patch(
        "app.api.routes.emotion.service.settings"
    ) as emotion_settings, patch("app.api.routes.transcription.service.settings") as transcription_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        emotion_settings.EMOTION_API_URL = "http://emotion:8000"
        transcription_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = FullAPIClient()

        with patch(
            "app.api.routes.full.service.transcription_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "text": "I have been waiting for hours and nothing was fixed.",
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "I have been waiting for hours and nothing was fixed.",
                        "speaker": "SPEAKER_00",
                    }
                ],
            },
        ), patch(
            "app.api.routes.full.service.emotion_client.analyze_bytes",
            new_callable=AsyncMock,
            return_value={
                "top_emotion": "neutral",
                "top_score": 0.9,
                "emotions": [{"label": "neutral", "score": 0.9}],
            },
        ):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["top_emotion"] == "frustrated"
    assert result["segments"][0]["emotion"] == "frustrated"


@pytest.mark.asyncio
async def test_full_service_analyze_local_file_falls_back_when_emotion_service_unreachable():
    with patch("app.api.routes.full.service.settings") as service_settings, patch(
        "app.api.routes.emotion.service.settings"
    ) as emotion_settings, patch("app.api.routes.transcription.service.settings") as transcription_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        emotion_settings.EMOTION_API_URL = "http://emotion:8000"
        transcription_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = True
        api = FullAPIClient()

        with patch(
            "app.api.routes.full.service.transcription_client.analyze_local_file",
            new_callable=AsyncMock,
            return_value={
                "text": "This is terrible and unacceptable.",
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "This is terrible and unacceptable.",
                        "speaker": "SPEAKER_00",
                    }
                ],
            },
        ), patch(
            "app.api.routes.full.service.emotion_client.analyze_local_file",
            new_callable=AsyncMock,
            side_effect=Exception("service unreachable"),
        ):
            result = await api.analyze_local_file("/tmp/clip.wav")

    assert result["top_emotion"] == "angry"
    assert result["segments"][0]["emotion"] == "angry"


@pytest.mark.asyncio
async def test_full_service_normalizes_remote_payload():
    with patch("app.api.routes.full.service.settings") as service_settings, patch(
        "app.core.kaggle_client.settings"
    ) as client_settings:
        service_settings.WHISPERX_API_URL = "http://whisperx:8000"
        client_settings.IS_LOCAL = False
        client_settings.KAGGLE_SERVER_URL = "https://kaggle.example.com"
        client_settings.KAGGLE_NGROK_URL = ""
        api = FullAPIClient()

        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "text": "hello",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": "UNKNOWN",
                    "emotion": "中立/neutral",
                    "emotion_scores": [{"label": "中立/neutral", "score": 0.95}],
                }
            ],
            "top_emotion": "中立/neutral",
            "emotions": [{"label": "中立/neutral", "score": 0.95}],
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            result = await api.analyze_bytes(b"audio", "clip.wav")

    assert result["top_emotion"] == "neutral"
    assert result["segments"][0]["emotion"] == "neutral"
