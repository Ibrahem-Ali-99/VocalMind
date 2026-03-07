
from unittest.mock import MagicMock, patch
from uuid import uuid4

def test_emotion_confidence_threshold_logic(client, mock_user):
    """
    Verifies that the backend correctly handles low-confidence emotion results
    if we were to implement a threshold filter (currently it returns all).
    """
    # This is a placeholder for actual model business logic validation
    pass

def test_agent_dispute_workflow_logic(client):
    """
    Verifies that the dispute endpoint correctly sets the flagging fields.
    """
    event_id = uuid4()
    # Create a stable mock client
    mock_client = MagicMock()
    
    # Use dependency_overrides for a stable mock across the request
    from app.api.deps import get_supabase
    client.app.dependency_overrides[get_supabase] = lambda: mock_client
    
    try:
        # Mock the auth check
        agent_id = str(uuid4())
        with patch("app.api.routes.emotion.dispute_router._get_current_user", return_value={"id": agent_id, "role": "agent", "organization_id": str(uuid4())}):
            # Mock the ownership check
            with patch("app.api.routes.emotion.dispute_router._get_event_and_assert_ownership", return_value={}):
                
                # Explicitly mock the chain to ensure it returns our data
                mock_execute = MagicMock()
                mock_execute.data = [{
                    "id": str(event_id),
                    "agent_flagged_by": agent_id,
                    "agent_flagged_at": "2024-03-06T12:00:00Z",
                    "agent_flag_note": "Wrong emotion"
                }]
                mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_execute
                
                response = client.post(
                    f"/api/v1/interactions/emotion-events/{event_id}/dispute?token=dummy-token",
                    json={"agent_flag_note": "Wrong emotion"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["is_flagged"] is True
                assert data["agent_flag_note"] == "Wrong emotion"
    finally:
        # Clean up (conftest.py also clears it, but being safe here)
        del client.app.dependency_overrides[get_supabase]
