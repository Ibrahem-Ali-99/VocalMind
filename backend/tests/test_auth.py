"""
Tests for auth routes and logic.

Covers Google OAuth redirect, invalid token handling, Google direct auth,
and password login logic using an injected Mock AsyncSession.
"""

from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.security import get_password_hash
from app.models.user import User as UserModel


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db_session(client: TestClient):
    """Overrides get_db with an AsyncMock session."""
    session = AsyncMock()
    # Default: select().where() returns empty result
    session.exec.return_value.first.return_value = None
    
    from app.api.deps import get_db
    async def _override_get_db():
        yield session

    client.app.dependency_overrides[get_db] = _override_get_db
    yield session
    client.app.dependency_overrides.pop(get_db, None)


def _fake_user(email="agent@test.com", password="correct", is_active=True) -> UserModel:
    return UserModel(
        id=uuid4(),
        organization_id=uuid4(),
        email=email,
        password_hash=get_password_hash(password) if password else None,
        name="Test",
        role="agent",
        is_active=is_active,
    )


def _mock_exec_results(mock_session, *results):
    """Queue up fetch results for sequential session.exec() calls."""
    mock_results = []
    for res in results:
        m = MagicMock()
        m.first.return_value = res
        mock_results.append(m)
    mock_session.exec.side_effect = mock_results


# ── Password Login Tests ─────────────────────────────────────────────────────

def test_login_valid_credentials(client: TestClient, mock_db_session: AsyncMock):
    _mock_exec_results(mock_db_session, _fake_user(password="right"))
    response = client.post("/api/v1/auth/login/access-token", data={"username": "agent@test.com", "password": "right"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_wrong_password(client: TestClient, mock_db_session: AsyncMock):
    _mock_exec_results(mock_db_session, _fake_user(password="right"))
    response = client.post("/api/v1/auth/login/access-token", data={"username": "agent@test.com", "password": "wrong"})
    assert response.status_code == 400


def test_login_nonexistent_user(client: TestClient, mock_db_session: AsyncMock):
    _mock_exec_results(mock_db_session, None)
    response = client.post("/api/v1/auth/login/access-token", data={"username": "ghost@test.com", "password": "any"})
    assert response.status_code == 400


def test_login_inactive_user(client: TestClient, mock_db_session: AsyncMock):
    _mock_exec_results(mock_db_session, _fake_user(is_active=False))
    response = client.post("/api/v1/auth/login/access-token", data={"username": "inactive@test.com", "password": "pw"})
    assert response.status_code == 400


# ── Google Direct Auth ───────────────────────────────────────────────────────

@patch("app.api.routes.auth.router.verify_google_token")
def test_google_direct_auth_existing_user(mock_verify, client: TestClient, mock_db_session: AsyncMock):
    mock_verify.return_value = MagicMock(email="test@google.com", name="G")
    _mock_exec_results(mock_db_session, _fake_user(email="test@google.com"))
    response = client.post("/api/v1/auth/google?token=good")
    assert response.status_code == 200
    assert "access_token" in response.json()


@patch("app.api.routes.auth.router.verify_google_token")
def test_google_direct_auth_new_user(mock_verify, client: TestClient, mock_db_session: AsyncMock):
    mock_verify.return_value = MagicMock(email="new@google.com", name="New")
    
    # 1. No user found. 2. Find existing Org (to skip org creation).
    org = MagicMock()
    org.id = uuid4()
    _mock_exec_results(mock_db_session, None, org)
    
    response = client.post("/api/v1/auth/google?token=good")
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_google_direct_invalid_token(client: TestClient):
    with patch("app.api.routes.auth.router.verify_google_token", return_value=None):
        response = client.post("/api/v1/auth/google?token=bad-token")
    assert response.status_code == 400


# ── Google Callback Auth ─────────────────────────────────────────────────────

def test_login_page_redirect(client: TestClient):
    response = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert response.status_code != 500


@pytest.mark.asyncio
async def test_google_callback_invalid_state(client: TestClient):
    response = client.get("/api/v1/auth/google/callback?code=123&state=bad")
    assert response.status_code == 400


@patch("app.api.routes.auth.router._oauth_states", {"good-state"})
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
@patch("app.api.routes.auth.router.verify_google_token")
def test_google_callback_success(mock_verify, mock_post, client: TestClient, mock_db_session: AsyncMock):
    # Mock token exchange
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id_token": "valid"}
    mock_post.return_value = mock_resp
    
    mock_verify.return_value = MagicMock(email="test@test.com", name="T")
    _mock_exec_results(mock_db_session, _fake_user())
    
    response = client.get("/api/v1/auth/google/callback?code=xyz&state=good-state", follow_redirects=False)
    assert response.status_code == 307
    assert "/login/success" in response.headers["location"]
    assert "vocalmind_token" in response.cookies


def test_token_validation_no_auth(client: TestClient):
    response = client.get("/api/v1/interactions/emotion-events/flagged?token=invalid_token")
    assert response.status_code == 403
