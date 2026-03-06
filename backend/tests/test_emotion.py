"""
Unit tests for the Emotion Analysis service and endpoints.

This module validates the handling of audio file uploads, extension checks,
and the integration with the remote Kaggle GPU worker (via mocking).
"""


from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

@patch("app.api.routes.emotion.router.emotion_client.analyze_local_file", new_callable=AsyncMock)
def test_analyze_local_file_success(mock_analyze, client: TestClient):
    """
    Tests successful emotion analysis using a mocked remote response.

    This test simulates a valid response from the Kaggle GPU worker
    without requiring an actual remote connection or GPU resources.

    Args:
        mock_analyze (AsyncMock): Mock for the API client's analysis method.
        client (TestClient): The FastAPI test client fixture.

    Asserts:
        Status code is 200.
        Response contains mocked emotion data.
    """
    # Configure the mock to return a sample emotion result
    mock_analyze.return_value = {
        "top_emotion": "happy",
        "top_score": 0.95,
        "emotions": [{"label": "happy", "score": 0.95}, {"label": "neutral", "score": 0.05}],
        "filename": "sample.wav"
    }

    # Execute request
    response = client.post(
        "/api/v1/emotion/analyze",
        json={"file_path": "dummy_path/sample.wav"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["top_emotion"] == "happy"
    assert data["top_score"] == 0.95
    assert len(data["emotions"]) == 2

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
        json={"file_path": "test_audio.txt"}
    )
    assert response.status_code == 400
    # Match the exact message from router.py: "Only .wav and .mp3 files are supported."
    assert "Only .wav and .mp3 files are supported" in response.json()["detail"]
