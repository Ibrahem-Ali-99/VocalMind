"""
Agent Emotion Event Dispute Router
===================================
Endpoints consumed by the Agent's Session Inspector to dispute LLM-generated
emotion events they believe are incorrect.

Workflow:
  1. Agent sees their emotion events in their own Session Inspector.
  2. Agent clicks "Dispute" on an event → POST /emotion-events/{event_id}/dispute
  3. is_flagged = TRUE, agent_flagged_by = agent_id, agent_flagged_at = now()
  4. Manager's review queue (GET /emotion-events/flagged) surfaces flagged events,
     showing which agent disputed each one and their optional note.
  5. Manager submits EmotionFeedback via the existing feedback endpoint — unchanged.
  6. Agent or manager can retract a dispute via DELETE /emotion-events/{event_id}/dispute
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List

from app.api.deps import get_supabase
from app.models.enums import UserRole

router = APIRouter(prefix="/emotion-events", tags=["emotion-events"])


# ── Request / Response Schemas ───────────────────────────────────────────────

class DisputeRequest(BaseModel):
    agent_flag_note: Optional[str] = None  # free-text "why I think this is wrong"


class DisputeResponse(BaseModel):
    event_id: UUID
    is_flagged: bool
    agent_flagged_by: UUID
    agent_flagged_at: datetime
    agent_flag_note: Optional[str]
    message: str


class FlaggedEventItem(BaseModel):
    """Summary of one agent-disputed event for the manager review queue."""
    event_id: UUID
    interaction_id: UUID
    previous_emotion: Optional[str]
    new_emotion: str
    llm_justification: Optional[str]
    jump_to_seconds: float
    confidence_score: Optional[float]
    # Agent dispute info
    agent_id: UUID
    agent_name: str
    agent_flagged_at: datetime
    agent_flag_note: Optional[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_current_user(supabase, token: str) -> dict:
    """Resolve JWT token to a user row. Raises 401 if invalid."""
    response = supabase.auth.get_user(token)
    if not response or not response.user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    user_id = response.user.id
    result = supabase.table("users").select("*").eq("id", user_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return result.data


async def _get_event_and_assert_ownership(supabase, event_id: UUID, agent_id: str) -> dict:
    """
    Fetch an emotion event and assert it belongs to an interaction
    that the calling agent actually handled (agent_id match on interaction).
    Raises 403 if the agent tries to flag someone else's event.
    """
    result = (
        supabase.table("emotion_events")
        .select("*, interactions!inner(agent_id)")
        .eq("id", str(event_id))
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Emotion event not found.")

    event = result.data
    interaction_agent = event.get("interactions", {}).get("agent_id")

    if interaction_agent != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only dispute emotion events from your own interactions.",
        )
    return event


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/{event_id}/dispute",
    response_model=DisputeResponse,
    status_code=status.HTTP_200_OK,
    summary="Agent disputes an emotion event",
    description=(
        "Marks an emotion event as agent-disputed. "
        "Sets is_flagged=TRUE and records who flagged it and when. "
        "The manager's review queue will surface this event. "
        "Only the agent who handled the interaction may dispute its events."
    ),
)
async def dispute_emotion_event(
    event_id: UUID,
    body: DisputeRequest,
    token: str,
    supabase=Depends(get_supabase),
):
    # 1. Authenticate
    user = await _get_current_user(supabase, token)

    if user["role"] != UserRole.agent:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only agents can dispute emotion events. Managers use the review queue.",
        )

    # 2. Fetch event and verify the agent owns this interaction
    await _get_event_and_assert_ownership(supabase, event_id, user["id"])

    # 3. Write the dispute
    now = datetime.now(timezone.utc).isoformat()
    update_payload = {
        "is_flagged":       True,
        "agent_flagged_by": user["id"],
        "agent_flagged_at": now,
        "agent_flag_note":  body.agent_flag_note,
    }

    result = (
        supabase.table("emotion_events")
        .update(update_payload)
        .eq("id", str(event_id))
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save dispute. Please try again.",
        )

    updated = result.data[0]
    return DisputeResponse(
        event_id=UUID(updated["id"]),
        is_flagged=True,
        agent_flagged_by=UUID(updated["agent_flagged_by"]),
        agent_flagged_at=datetime.fromisoformat(updated["agent_flagged_at"]),
        agent_flag_note=updated.get("agent_flag_note"),
        message=f"Event disputed successfully. Your manager will be notified in their review queue.",
    )


@router.delete(
    "/{event_id}/dispute",
    status_code=status.HTTP_200_OK,
    summary="Agent retracts their dispute",
    description=(
        "Removes the agent's dispute from an emotion event. "
        "Resets is_flagged, agent_flagged_by, agent_flagged_at, agent_flag_note to their defaults. "
        "Only the agent who originally flagged the event can retract it, "
        "unless a manager clears it after review."
    ),
)
async def retract_dispute(
    event_id: UUID,
    token: str,
    supabase=Depends(get_supabase),
):
    user = await _get_current_user(supabase, token)

    # Fetch event
    result = (
        supabase.table("emotion_events")
        .select("id, agent_flagged_by, is_flagged")
        .eq("id", str(event_id))
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Emotion event not found.")

    event = result.data

    # Agent can only retract their OWN dispute; manager can clear any
    if user["role"] == UserRole.agent:
        if event.get("agent_flagged_by") != user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only retract disputes you submitted.",
            )

    clear_payload = {
        "is_flagged":       False,
        "agent_flagged_by": None,
        "agent_flagged_at": None,
        "agent_flag_note":  None,
    }

    supabase.table("emotion_events").update(clear_payload).eq("id", str(event_id)).execute()
    return {"event_id": str(event_id), "message": "Dispute retracted."}


@router.get(
    "/flagged",
    response_model=List[FlaggedEventItem],
    summary="Manager: get all agent-flagged emotion events for their org",
    description=(
        "Returns all emotion events that agents have disputed, "
        "enriched with the agent's name and their optional note. "
        "Scoped to the calling manager's organization. "
        "Manager-only endpoint."
    ),
)
async def get_flagged_events(
    token: str,
    supabase=Depends(get_supabase),
):
    user = await _get_current_user(supabase, token)

    if user["role"] != UserRole.manager:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers can access the flagged events review queue.",
        )

    org_id = user["organization_id"]

    # Fetch flagged events joined with interaction (for org scoping) and the agent's name
    result = (
        supabase.table("emotion_events")
        .select(
            "id, interaction_id, previous_emotion, new_emotion, "
            "llm_justification, jump_to_seconds, confidence_score, "
            "agent_flagged_by, agent_flagged_at, agent_flag_note, "
            "interactions!inner(organization_id, agent_id), "
            "users!emotion_events_agent_flagged_by_fkey(id, name)"
        )
        .eq("interactions.organization_id", org_id)
        .eq("is_flagged", True)
        .not_.is_("agent_flagged_by", "null")
        .order("agent_flagged_at", desc=True)
        .execute()
    )

    items: List[FlaggedEventItem] = []
    for row in result.data or []:
        agent_info = row.get("users") or {}
        items.append(
            FlaggedEventItem(
                event_id=UUID(row["id"]),
                interaction_id=UUID(row["interaction_id"]),
                previous_emotion=row.get("previous_emotion"),
                new_emotion=row["new_emotion"],
                llm_justification=row.get("llm_justification"),
                jump_to_seconds=row["jump_to_seconds"],
                confidence_score=row.get("confidence_score"),
                agent_id=UUID(row["agent_flagged_by"]),
                agent_name=agent_info.get("name", "Unknown Agent"),
                agent_flagged_at=datetime.fromisoformat(row["agent_flagged_at"]),
                agent_flag_note=row.get("agent_flag_note"),
            )
        )

    return items
