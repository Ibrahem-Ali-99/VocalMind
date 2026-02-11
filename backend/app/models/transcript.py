from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4

class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id", unique=True)
    full_text: str
    confidence_score: Optional[float] = None
