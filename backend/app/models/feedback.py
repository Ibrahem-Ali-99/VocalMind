from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4


class EmotionFeedback(SQLModel, table=True):
    """Manager correction on an emotion event."""
    __tablename__ = "emotion_feedback"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    emotion_event_id: UUID = Field(foreign_key="emotion_events.id")
    provided_by_user_id: UUID = Field(foreign_key="users.id")
    llm_justification: Optional[str] = None  # copied from the emotion event at feedback time
    corrected_emotion: str = Field(max_length=50)
    corrected_justification: Optional[str] = None
    correction_reason: Optional[str] = None
    is_used_in_training: bool = Field(default=False)


class ComplianceFeedback(SQLModel, table=True):
    """Manager correction on a policy compliance record."""
    __tablename__ = "compliance_feedback"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    policy_compliance_id: UUID = Field(foreign_key="policy_compliance.id")
    provided_by_user_id: UUID = Field(foreign_key="users.id")
    original_is_compliant: bool
    corrected_is_compliant: bool
    original_score: Optional[float] = None
    corrected_score: Optional[float] = None
    correction_reason: Optional[str] = None
    is_used_in_training: bool = Field(default=False)
