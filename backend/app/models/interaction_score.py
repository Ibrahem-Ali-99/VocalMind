from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID, uuid4


class InteractionScore(SQLModel, table=True):
    """One score record per interaction (1:1)."""
    __tablename__ = "interaction_scores"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id", unique=True)
    overall_score: Optional[float] = None  # 0.0–10.0
    empathy_score: Optional[float] = None
    policy_score: Optional[float] = None
    resolution_score: Optional[float] = None
    total_silence_duration_seconds: Optional[float] = None
    average_response_time_seconds: Optional[float] = None
    was_resolved: Optional[bool] = None
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
