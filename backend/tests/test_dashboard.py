"""
Unit tests for the dashboard stats endpoint.

Strategy: Pre-populate the cache with known data to bypass PostgreSQL-specific
SQL and test the endpoint response structure cleanly.
"""
import pytest
from uuid import UUID

from app.api.deps import get_current_user
from app.core.cache import dashboard_cache
from app.models.enums import UserRole
from app.models.user import User


TEST_ORG_ID = UUID("8c662e42-9200-493e-aeb2-17d23ac7222a")
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture(autouse=True)
def override_current_user(client):
    async def _override_current_user():
        return User(
            id=TEST_USER_ID,
            organization_id=TEST_ORG_ID,
            email="manager@nexalink.com",
            password_hash="test_hash",
            name="Test Manager",
            role=UserRole.manager,
            is_active=True,
        )

    client.app.dependency_overrides[get_current_user] = _override_current_user
    yield
    client.app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Fixture: canonical dashboard payload (mirrors what the real route returns)
# ---------------------------------------------------------------------------
MOCK_DASHBOARD_DATA = {
    "kpis": {
        "avgScore": 7.5,
        "totalCalls": 42,
        "resolutionRate": 80.0,
        "violationCount": 3,
    },
    "weeklyTrend": [
        {"day": "Mon", "score": 7},
        {"day": "Tue", "score": 8},
    ],
    "emotionDistribution": [
        {"name": "Neutral", "value": 60, "color": "#6B7280"},
        {"name": "Happy", "value": 40, "color": "#10B981"},
    ],
    "policyCompliance": [
        {"category": "Greeting", "rate": 85, "color": "#3B82F6"},
    ],
    "agentPerformance": [
        {
            "name": "Alice",
            "empathy": 8,
            "policy": 9,
            "resolution": 7,
            "overallScore": 8,
            "trend": "up",
        }
    ],
    "interactions": [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "agentName": "Alice",
            "date": "2025-01-01",
            "time": "09:00 AM",
            "duration": "5:30",
            "language": "en",
            "overallScore": 8,
            "empathyScore": 8,
            "policyScore": 9,
            "resolutionScore": 7,
            "resolved": True,
            "hasViolation": False,
            "hasOverlap": False,
        }
    ],
}


@pytest.fixture(autouse=True)
def seed_cache():
    """Pre-load the cache so every test bypasses the real DB."""
    dashboard_cache.set(f"manager_stats_{TEST_ORG_ID}", MOCK_DASHBOARD_DATA)
    yield
    dashboard_cache.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_dashboard_stats_returns_200(client):
    """Dashboard stats endpoint should return 200 when cache is warm."""
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200, response.text


def test_get_dashboard_stats_has_required_keys(client):
    """Response must include all top-level dashboard keys."""
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    data = response.json()

    required_keys = [
        "kpis",
        "weeklyTrend",
        "emotionDistribution",
        "policyCompliance",
        "agentPerformance",
        "interactions",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"


def test_get_dashboard_stats_kpis_structure(client):
    """KPI sub-object must include the four expected numeric fields."""
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    kpis = response.json()["kpis"]

    assert "avgScore" in kpis
    assert "totalCalls" in kpis
    assert "resolutionRate" in kpis
    assert "violationCount" in kpis


def test_get_dashboard_stats_kpis_values(client):
    """KPI values should match what was stored in cache."""
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    kpis = response.json()["kpis"]

    assert kpis["avgScore"] == 7.5
    assert kpis["totalCalls"] == 42
    assert kpis["resolutionRate"] == 80.0
    assert kpis["violationCount"] == 3


def test_get_dashboard_stats_returns_cached_data(client):
    """Two requests return identical data (both served from cache)."""
    first = client.get("/api/v1/dashboard/stats")
    second = client.get("/api/v1/dashboard/stats")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_get_dashboard_stats_cache_is_used(client):
    """Cache should still be populated after the request completes."""
    client.get("/api/v1/dashboard/stats")
    assert dashboard_cache.get(f"manager_stats_{TEST_ORG_ID}") is not None


def test_get_dashboard_stats_interactions_structure(client):
    """Each interaction entry should have required fields."""
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    interactions = response.json()["interactions"]

    assert len(interactions) >= 1
    interaction = interactions[0]
    for field in ["id", "agentName", "date", "duration", "overallScore", "resolved"]:
        assert field in interaction, f"Missing interaction field: {field}"
