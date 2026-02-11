from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import JSON, Column, Enum as SAEnum
from app.models.enums import FeedbackType

class HumanFeedback(SQLModel, table=True):
    __tablename__ = "human_feedback"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    provided_by_user_id: UUID = Field(foreign_key="users.id")
    feedback_type: FeedbackType = Field(sa_type=SAEnum(FeedbackType, name="feedback_type_enum", create_constraint=False, native_enum=True))
    ai_output: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    corrected_output: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    correction_reason: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
