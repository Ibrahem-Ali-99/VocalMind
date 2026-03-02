from datetime import datetime
from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import SpeakerRole


class EmotionEvent(SQLModel, table=True):
    """AI-generated emotion shift event.

    Agent-flagging workflow (v5.2):
      - Agents see their own events in the Session Inspector.
      - If an agent thinks an event is wrong they click "Dispute" —
        this sets agent_flagged_by + agent_flagged_at + (optional) agent_flag_note.
      - is_flagged is set to TRUE at the same time as a quick boolean index.
      - The manager's review queue filters on is_flagged=TRUE and shows which
        agent disputed the event and why.
      - The manager may then submit an EmotionFeedback correction — that
        RLHF workflow is unchanged.
    """
    __tablename__ = "emotion_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    utterance_id: UUID = Field(foreign_key="utterances.id")
    previous_emotion: Optional[str] = Field(default=None, max_length=50)
    new_emotion: str = Field(max_length=50)
    emotion_delta: Optional[float] = None
    speaker_role: SpeakerRole = Field(
        sa_type=SAEnum(SpeakerRole, name="speaker_role_enum", create_constraint=False, native_enum=True),
    )
    llm_justification: Optional[str] = None      # AI-generated causal explanation
    jump_to_seconds: float                        # exact second in audio (for waveform jump button)
    confidence_score: Optional[float] = None      # 0.0–1.0

    # ── Manager-level flag ──────────────────────────────────
    # TRUE when an agent has disputed this event (set together with agent_flagged_by).
    # Also set TRUE by the manager directly when reviewing without an agent prompt.
    is_flagged: bool = Field(default=False)

    # ── Agent-dispute fields (v5.2) ─────────────────────────
    # Populated when the agent clicks "Dispute" in their Session Inspector.
    # NULL means no agent has disputed this event.
    agent_flagged_by: Optional[UUID] = Field(default=None, foreign_key="users.id")
    agent_flagged_at: Optional[datetime] = Field(default=None)
    agent_flag_note:  Optional[str] = Field(default=None)  # agent's optional explanation
