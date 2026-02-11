from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import ProcessingStatus

class Interaction(SQLModel, table=True):
    __tablename__ = "interactions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    agent_id: UUID = Field(foreign_key="agents.id")
    audio_file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[int] = None
    file_format: Optional[str] = None
    interaction_date: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.pending, sa_type=SAEnum(ProcessingStatus, name="processing_status_enum", create_constraint=False, native_enum=True))
    language_detected: Optional[str] = None
    has_overlap: Optional[bool] = Field(default=False)
