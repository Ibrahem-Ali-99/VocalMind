from sqlmodel import SQLModel, Field
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

class HumanFeedback(SQLModel, table=True):
    """Generic human feedback on AI outputs."""
    __tablename__ = "human_feedback"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    provided_by_user_id: UUID = Field(foreign_key="users.id")
    feedback_type: str = Field(max_length=50)  # e.g., 'emotion_label', 'score', 'compliance'
    ai_output: Dict[str, Any] = Field(default_factory=dict, sa_type="JSONB")
    corrected_output: Dict[str, Any] = Field(default_factory=dict, sa_type="JSONB")
    correction_reason: Optional[str] = None
