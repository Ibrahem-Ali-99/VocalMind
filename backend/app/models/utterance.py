from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import SpeakerRole


class Utterance(SQLModel, table=True):
    __tablename__ = "utterances"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    transcript_id: UUID = Field(foreign_key="transcripts.id")
    speaker_role: SpeakerRole = Field(
        sa_type=SAEnum(SpeakerRole, name="speaker_role_enum", create_constraint=False, native_enum=True),
    )
    user_id: Optional[UUID] = Field(default=None, foreign_key="users.id")  # set if speaker is a known agent
    sequence_index: int  # ordering index for reconstructing dialogue
    start_time_seconds: float
    end_time_seconds: float
    text: str  # transcribed text from Whisper
    emotion: Optional[str] = Field(default=None, max_length=50)  # acoustic emotion label
    emotion_confidence: Optional[float] = None  # 0.0–1.0
