from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4


class Transcript(SQLModel, table=True):
    """One row per interaction (1:1)."""
    __tablename__ = "transcripts"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id", unique=True)
    full_text: Optional[str] = None  # computed from utterances after processing
    overall_confidence: Optional[float] = None  # renamed from confidence_score
