from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import SpeakerRole


class EmotionEvent(SQLModel, table=True):
    """All fields are AI-generated."""
    __tablename__ = "emotion_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    utterance_id: UUID = Field(foreign_key="utterances.id")  # now required, not optional
    previous_emotion: Optional[str] = Field(default=None, max_length=50)
    new_emotion: str = Field(max_length=50)
    emotion_delta: Optional[float] = None
    speaker_role: SpeakerRole = Field(
        sa_type=SAEnum(SpeakerRole, name="speaker_role_enum", create_constraint=False, native_enum=True),
    )
    llm_justification: Optional[str] = None  # AI-generated causal explanation
    timestamp_seconds: float  # exact second in audio
    confidence_score: Optional[float] = None  # 0.0–1.0
