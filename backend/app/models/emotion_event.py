from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import EventType

class EmotionEvent(SQLModel, table=True):
    __tablename__ = "emotion_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    utterance_id: Optional[UUID] = Field(default=None, foreign_key="utterances.id")
    event_type: EventType = Field(sa_type=SAEnum(EventType, name="event_type_enum", create_constraint=False, native_enum=True))
    previous_emotion: Optional[str] = None
    new_emotion: Optional[str] = None
    emotion_delta: Optional[float] = None
    trigger_category: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    speaker_role: Optional[str] = None
    verified_by_user_id: Optional[UUID] = Field(default=None, foreign_key="users.id")
