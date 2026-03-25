from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


@patch("app.api.routes.diarization.router.diarization_client.analyze_audio", new_callable=AsyncMock)
def test_diarization_analyze_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}
    response = client.post(
        "/api/v1/diarization/analyze",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["segments"][0]["speaker"] == "SPEAKER_00"


@patch("app.api.routes.diarization.router.diarization_client.analyze_local_file", new_callable=AsyncMock)
def test_diarization_analyze_local_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}
    response = client.post("/api/v1/diarization/analyze-local", json={"file_path": "call.wav"})
    assert response.status_code == 200
    assert response.json()["segments"][0]["speaker"] == "SPEAKER_00"


@patch("app.api.routes.transcription.router.transcription_client.analyze_audio", new_callable=AsyncMock)
def test_transcription_analyze_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"text": "hello", "language": "en", "segments": []}
    response = client.post(
        "/api/v1/transcription/analyze",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello"


@patch("app.api.routes.transcription.router.transcription_client.analyze_local_file", new_callable=AsyncMock)
def test_transcription_analyze_local_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"text": "hello", "language": "en", "segments": []}
    response = client.post("/api/v1/transcription/analyze-local", json={"file_path": "sample.wav"})
    assert response.status_code == 200
    assert response.json()["language"] == "en"


@patch("app.api.routes.vad.router.vad_client.analyze_audio", new_callable=AsyncMock)
def test_vad_analyze_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"speech_segments": [{"start": 0.0, "end": 1.0}]}
    response = client.post(
        "/api/v1/vad/analyze",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["speech_segments"][0]["end"] == 1.0


@patch("app.api.routes.vad.router.vad_client.analyze_local_file", new_callable=AsyncMock)
def test_vad_analyze_local_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {"speech_segments": [{"start": 0.0, "end": 1.0}]}
    response = client.post("/api/v1/vad/analyze-local", json={"file_path": "sample.wav"})
    assert response.status_code == 200
    assert response.json()["speech_segments"][0]["start"] == 0.0


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


@patch("app.api.routes.full.router.full_client.analyze_local_file", new_callable=AsyncMock)
def test_full_analyze_local_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {
        "text": "hello",
        "language": "en",
        "segments": [],
        "top_emotion": "neutral",
        "top_score": 0.88,
        "emotions": [{"label": "neutral", "score": 0.88}],
    }
    response = client.post("/api/v1/full/analyze-local", json={"file_path": "sample.wav"})
    assert response.status_code == 200
    assert response.json()["top_score"] == 0.88


def test_new_inference_routes_reject_invalid_extension(client: TestClient):
    for path in (
        "/api/v1/diarization/analyze",
        "/api/v1/transcription/analyze",
        "/api/v1/vad/analyze",
        "/api/v1/full/analyze",
    ):
        response = client.post(path, files={"file": ("notes.txt", b"nope", "text/plain")})
        assert response.status_code == 400
