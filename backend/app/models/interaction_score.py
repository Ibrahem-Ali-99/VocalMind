from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID

class InteractionScore(SQLModel, table=True):
    __tablename__ = "interaction_scores"

    interaction_id: UUID = Field(primary_key=True, foreign_key="interactions.id")
    overall_score: Optional[float] = None
    policy_score: Optional[float] = None
    total_silence_duration_seconds: Optional[float] = None
    average_response_time_seconds: Optional[float] = None
    was_resolved: Optional[bool] = None
