"""
Unit tests for the Emotion Analysis endpoints.

Covers /analyze (success, .mp3, invalid extension), /analyze-local (success), and /process (success, invalid extension).
"""

from uuid import uuid4
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


# ── /analyze ─────────────────────────────────────────────────────────────────

@patch("app.api.routes.emotion.router.emotion_client.analyze_audio", new_callable=AsyncMock)
def test_analyze_upload_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {
        "emotion": "happy",
        "confidence": 0.95,
        "raw_result": {"labels": ["happy", "neutral"], "scores": [0.95, 0.05]},
    }
    response = client.post(
        "/api/v1/emotion/analyze",
        files={"file": ("sample.wav", b"fake audio content", "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "happy"
    assert data["confidence"] == 0.95


@patch("app.api.routes.emotion.router.emotion_client.analyze_audio", new_callable=AsyncMock)
def test_analyze_mp3_accepted(mock_analyze, client: TestClient):
    """MP3 files should pass the extension check."""
    mock_analyze.return_value = {"emotion": "neutral", "confidence": 0.7}
    response = client.post(
        "/api/v1/emotion/analyze",
        files={"file": ("recording.mp3", b"fake mp3", "audio/mpeg")},
    )
    assert response.status_code == 200


def test_analyze_invalid_extension(client: TestClient):
    response = client.post(
        "/api/v1/emotion/analyze",
        files={"file": ("test_audio.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .wav and .mp3 files are supported" in response.json()["detail"]


# ── /analyze-local ───────────────────────────────────────────────────────────

@patch("app.api.routes.emotion.router.emotion_client.analyze_local_file", new_callable=AsyncMock)
def test_analyze_local_success(mock_analyze, client: TestClient):
    mock_analyze.return_value = {
        "top_emotion": "happy",
        "top_score": 0.95,
        "emotions": [{"label": "happy", "score": 0.95}, {"label": "neutral", "score": 0.05}],
        "filename": "sample.wav"
    }

    response = client.post(
        "/api/v1/emotion/analyze-local",
        json={"file_path": "dummy_path/sample.wav"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["top_emotion"] == "happy"


# ── /process ─────────────────────────────────────────────────────────────────

@patch("app.api.routes.emotion.router.process_audio", new_callable=AsyncMock)
def test_process_endpoint_success(mock_pipeline, client: TestClient):
    interaction_id = uuid4()
    mock_pipeline.return_value = [
        {"emotion": "angry", "start_time_seconds": 0.0, "end_time_seconds": 2.0},
    ]
    response = client.post(
        f"/api/v1/emotion/process?interaction_id={interaction_id}",
        files={"file": ("call.wav", b"audio-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_segments"] == 1
    assert data["interaction_id"] == str(interaction_id)


def test_process_invalid_extension(client: TestClient):
    interaction_id = uuid4()
    response = client.post(
        f"/api/v1/emotion/process?interaction_id={interaction_id}",
        files={"file": ("notes.pdf", b"not audio", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Only .wav and .mp3 files are supported" in response.json()["detail"]
