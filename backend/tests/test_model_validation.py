"""
Unit tests for the Dispute Router — agent dispute, retract, and flagged events.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.api.deps import get_supabase


# ── Helpers ──────────────────────────────────────────────────────────────────

def _override_supabase(client, mock_client):
    """Inject a mock Supabase client."""
    client.app.dependency_overrides[get_supabase] = lambda: mock_client


def _mock_supabase_update(mock_client, data):
    """Wire the table().update().eq().execute() chain."""
    mock_execute = MagicMock()
    mock_execute.data = data
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_execute


def _mock_supabase_select_single(mock_client, data):
    """Wire the table().select().eq().single().execute() chain."""
    mock_execute = MagicMock()
    mock_execute.data = data
    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_execute


# ── Dispute creation ────────────────────────────────────────────────────────

def test_agent_dispute_workflow_success(client):
    """Agents can dispute their own emotion events."""
    event_id = uuid4()
    agent_id = str(uuid4())
    mock_client = MagicMock()
    _override_supabase(client, mock_client)

    try:
        with (
            patch("app.api.routes.emotion.dispute_router._get_current_user",
                  return_value={"id": agent_id, "role": "agent", "organization_id": str(uuid4())}),
            patch("app.api.routes.emotion.dispute_router._get_event_and_assert_ownership",
                  return_value={}),
        ):
            _mock_supabase_update(mock_client, [{
                "id": str(event_id),
                "agent_flagged_by": agent_id,
                "agent_flagged_at": "2024-03-06T12:00:00Z",
                "agent_flag_note": "Wrong emotion",
            }])

            response = client.post(
                f"/api/v1/interactions/emotion-events/{event_id}/dispute?token=dummy",
                json={"agent_flag_note": "Wrong emotion"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["is_flagged"] is True
            assert data["agent_flag_note"] == "Wrong emotion"
    finally:
        del client.app.dependency_overrides[get_supabase]


def test_dispute_non_agent_rejected(client):
    """Managers cannot use the dispute endpoint."""
    mock_client = MagicMock()
    _override_supabase(client, mock_client)

    try:
        with patch("app.api.routes.emotion.dispute_router._get_current_user",
                    return_value={"id": str(uuid4()), "role": "manager", "organization_id": str(uuid4())}):
            response = client.post(
                f"/api/v1/interactions/emotion-events/{uuid4()}/dispute?token=t",
                json={},
            )
            assert response.status_code == 403
            assert "Only agents" in response.json()["detail"]
    finally:
        del client.app.dependency_overrides[get_supabase]


# ── Retract dispute ─────────────────────────────────────────────────────────

def test_retract_dispute_success(client):
    """Agent can retract their own dispute."""
    event_id = uuid4()
    agent_id = str(uuid4())
    mock_client = MagicMock()
    _override_supabase(client, mock_client)

    try:
        with patch("app.api.routes.emotion.dispute_router._get_current_user",
                    return_value={"id": agent_id, "role": "agent"}):
            _mock_supabase_select_single(mock_client, {
                "id": str(event_id),
                "agent_flagged_by": agent_id,
                "is_flagged": True,
            })

            response = client.delete(
                f"/api/v1/interactions/emotion-events/{event_id}/dispute?token=t",
            )
            assert response.status_code == 200
            assert "retracted" in response.json()["message"].lower()
    finally:
        del client.app.dependency_overrides[get_supabase]


def test_retract_dispute_wrong_agent(client):
    """Agent cannot retract another agent's dispute."""
    event_id = uuid4()
    mock_client = MagicMock()
    _override_supabase(client, mock_client)

    try:
        with patch("app.api.routes.emotion.dispute_router._get_current_user",
                    return_value={"id": str(uuid4()), "role": "agent"}):
            _mock_supabase_select_single(mock_client, {
                "id": str(event_id),
                "agent_flagged_by": str(uuid4()),  # different agent
                "is_flagged": True,
            })

            response = client.delete(
                f"/api/v1/interactions/emotion-events/{event_id}/dispute?token=t",
            )
            assert response.status_code == 403
            assert "only retract disputes you submitted" in response.json()["detail"].lower()
    finally:
        del client.app.dependency_overrides[get_supabase]


# ── Flagged events ───────────────────────────────────────────────────────────

def test_get_flagged_events_non_manager_rejected(client):
    """Non-managers get 403 on the flagged events queue."""
    mock_client = MagicMock()
    _override_supabase(client, mock_client)

    try:
        with patch("app.api.routes.emotion.dispute_router._get_current_user",
                    return_value={"id": str(uuid4()), "role": "agent", "organization_id": str(uuid4())}):
            response = client.get("/api/v1/interactions/emotion-events/flagged?token=t")
            assert response.status_code == 403
            assert "Only managers" in response.json()["detail"]
    finally:
        del client.app.dependency_overrides[get_supabase]
