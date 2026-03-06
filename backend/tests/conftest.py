"""
Shared pytest fixtures and configuration for the VocalMind Backend.

This module provides common fixtures for testing FastAPI endpoints, including
a pre-configured TestClient and mocked database sessions to ensure tests
remain isolated from production data.
"""

import pytest
from typing import Generator
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, SQLModel

# Mock database creation to avoid lifespan errors
with patch("app.core.database.create_db_and_tables", return_value=None):
    from app.main import app
from app.api.deps import get_session, get_supabase, get_db

# --- Fixtures ---

@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """Provides a functional SQLite in-memory Session for testing."""
    engine = create_engine(
        "sqlite:///:memory:", 
        connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture(name="client", autouse=True)
def client_fixture(session: Session) -> Generator[TestClient, None, None]:
    """
    Provides a FastAPI TestClient with database and supabase dependencies overriden.
    """
    async def _get_session_override():
        yield session
    
    def _get_supabase_override():
        return MagicMock()

    app.dependency_overrides[get_session] = _get_session_override
    app.dependency_overrides[get_db] = _get_session_override
    app.dependency_overrides[get_supabase] = _get_supabase_override
    
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

@pytest.fixture(name="mock_user")
def mock_user_fixture() -> dict:
    """Provides a sample mock user dictionary for authentication tests."""
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "email": "test@vocalmind.ai",
        "full_name": "Test User",
        "role": "manager"
    }
