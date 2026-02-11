from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import SpeakerRole

class Utterance(SQLModel, table=True):
    __tablename__ = "utterances"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    speaker_role: SpeakerRole = Field(sa_type=SAEnum(SpeakerRole, name="speaker_role_enum", create_constraint=False, native_enum=True))
    start_time_seconds: float
    end_time_seconds: float
    emotion_label: Optional[str] = None
    emotion_confidence: Optional[float] = None
