"""
Unit tests for the core API endpoints of the VocalMind Backend.

This module contains tests for the root ("/") and health check ("/health")
endpoints to ensure basic availability and routing are functional.
"""

from fastapi.testclient import TestClient

def test_read_root(client: TestClient):
    """
    Tests the root GET endpoint for a welcome message.

    Args:
        client (TestClient): The FastAPI test client fixture.

    Asserts:
        Status code is 200.
        Response contains the expected welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to VocalMind API"}

def test_health_check(client: TestClient):
    """
    Tests the health check endpoint for system status.

    Args:
        client (TestClient): The FastAPI test client fixture.

    Asserts:
        Status code is 200.
        Response status is "ok".
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
