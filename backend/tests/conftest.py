"""
Shared pytest fixtures and configuration for the VocalMind Backend.

This module provides common fixtures for testing FastAPI endpoints, including
a pre-configured TestClient and mocked database sessions to ensure tests
remain isolated from production data.
"""

import pytest
import importlib
from typing import Generator
from unittest.mock import AsyncMock, MagicMock
from app.api.deps import get_session, get_supabase, get_db
from app.core.config import settings
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, SQLModel

# Stub startup/lifespan hooks to keep tests isolated and fast.
app_main = importlib.import_module("app.main")
app_main.create_db_and_tables = AsyncMock(return_value=None)
app_main.prewarm_dashboard_cache = AsyncMock(return_value=None)
app_main.seed_nexalink_main = AsyncMock(return_value=None)
app_main.start_processing_worker = AsyncMock(return_value=None)
app_main.stop_processing_worker = AsyncMock(return_value=None)
app = app_main.app
settings.SECRET_KEY = "test-secret-key-minimum-32-bytes-long"

# --- Fixtures ---

@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """Provides a functional SQLite in-memory Session for testing."""
    # Import ALL models to ensure they are registered with SQLModel.metadata
    # before create_all is called — otherwise tables won't be created.
    import app.models  # noqa: F401 — triggers all sub-imports via __init__.py
    
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
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
