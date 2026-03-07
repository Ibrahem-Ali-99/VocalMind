"""
Unit tests for the Emotion Analysis service and endpoints.

This module validates the handling of audio file uploads, extension checks,
and the integration with the remote Kaggle GPU worker (via mocking).
"""


from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

@patch("app.api.routes.emotion.router.emotion_client.analyze_audio", new_callable=AsyncMock)
def test_analyze_upload_success(mock_analyze, client: TestClient):
    """
    Tests successful emotion analysis using a mocked remote response.
    """
    # Configure the mock to return a sample emotion result (matching Kaggle format)
    mock_analyze.return_value = {
        "emotion": "happy",
        "confidence": 0.95,
        "raw_result": {
            "labels": ["happy", "neutral"],
            "scores": [0.95, 0.05]
        }
    }

    # Execute request with multipart file
    response = client.post(
        "/api/v1/emotion/analyze",
        files={"file": ("sample.wav", b"fake audio content", "audio/wav")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "happy"
    assert data["confidence"] == 0.95

def test_analyze_invalid_extension(client: TestClient):
    """
    Tests that the API rejects files with unsupported extensions.

    Args:
        client (TestClient): The FastAPI test client fixture.

    Asserts:
        Status code is 400.
        Detail message specifies invalid file format.
    """
    response = client.post(
        "/api/v1/emotion/analyze",
        files={"file": ("test_audio.txt", b"not audio", "text/plain")}
    )
    assert response.status_code == 400
    # Match the exact message from router.py: "Only .wav and .mp3 files are supported."
    assert "Only .wav and .mp3 files are supported" in response.json()["detail"]
