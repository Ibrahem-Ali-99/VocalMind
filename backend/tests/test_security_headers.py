
from fastapi.testclient import TestClient

def test_security_headers_present(client: TestClient):
    """
    Verifies that essential security headers are present in the response.
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    # Check for basic security headers (FastAPI/Starlette defaults or custom)
    # Note: Modern browsers and FastAPI/Starlette often provide these or they are added via middleware
    headers = response.headers
    
    # We check for common ones that should be present in a hardened app
    # If these are missing, we might need to add them via middleware in app/main.py
    assert "x-content-type-options" in headers
    assert headers["x-content-type-options"] == "nosniff"
    
    assert "x-frame-options" in headers
    assert headers["x-frame-options"] in ["DENY", "SAMEORIGIN"]
