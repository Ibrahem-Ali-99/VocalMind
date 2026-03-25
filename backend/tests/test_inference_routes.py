from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


@patch("app.api.routes.transcription.router.transcription_client.analyze_audio", new_callable=AsyncMock)
def test_transcription_analyze_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"text": "hello", "language": "en", "segments": []}
    response = client.post(
        "/api/v1/transcription/analyze",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello"


@patch("app.api.routes.vad.router.vad_client.analyze_audio", new_callable=AsyncMock)
def test_vad_analyze_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"speech_segments": [{"start": 0.0, "end": 1.0}]}
    response = client.post(
        "/api/v1/vad/analyze",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["speech_segments"][0]["end"] == 1.0


@patch("app.api.routes.full.router.full_client.analyze_audio", new_callable=AsyncMock)
def test_full_analyze_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {
        "text": "hello",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "UNKNOWN", "emotion": "neutral", "emotion_scores": []}],
        "top_emotion": "neutral",
        "top_score": 0.88,
        "emotions": [{"label": "neutral", "score": 0.88}],
    }
    response = client.post(
        "/api/v1/full/analyze",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["top_emotion"] == "neutral"


def test_new_inference_routes_reject_invalid_extension(client: TestClient):
    for path in (
        "/api/v1/transcription/analyze",
        "/api/v1/vad/analyze",
        "/api/v1/full/analyze",
    ):
        response = client.post(path, files={"file": ("notes.txt", b"nope", "text/plain")})
        assert response.status_code == 400
