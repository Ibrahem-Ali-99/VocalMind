"""Tests for app.api.routes.auth.service — Google token verification."""

from unittest.mock import patch

from app.api.routes.auth.service import verify_google_token


@patch("app.api.routes.auth.service.id_token.verify_oauth2_token")
def test_valid_token(mock_verify):
    mock_verify.return_value = {"email": "a@b.com", "name": "Alice", "picture": "https://pic"}
    result = verify_google_token("good")
    assert result is not None
    assert result.email == "a@b.com"
    assert result.name == "Alice"


@patch("app.api.routes.auth.service.id_token.verify_oauth2_token")
def test_expired_token_returns_none(mock_verify):
    mock_verify.side_effect = ValueError("Token expired")
    assert verify_google_token("expired") is None


@patch("app.api.routes.auth.service.id_token.verify_oauth2_token")
def test_unexpected_error_returns_none(mock_verify):
    mock_verify.side_effect = RuntimeError("Network")
    assert verify_google_token("bad") is None
