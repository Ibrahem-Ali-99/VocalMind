from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, SmallInteger, BigInteger, Enum as SAEnum
from app.models.enums import ProcessingStatus


class Interaction(SQLModel, table=True):
    __tablename__ = "interactions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    agent_id: UUID = Field(foreign_key="users.id")  # FK → users (merged)
    audio_file_path: str
    file_size_bytes: int = Field(sa_column=Column(BigInteger, nullable=False))
    duration_seconds: int
    file_format: str = Field(max_length=10)
    interaction_date: datetime
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.pending,
        sa_type=SAEnum(ProcessingStatus, name="processing_status_enum", create_constraint=False, native_enum=True),
    )
    language_detected: Optional[str] = Field(default=None, max_length=10)
    has_overlap: bool = Field(default=False)
    channel_count: int = Field(default=1, sa_column=Column(SmallInteger, nullable=False, server_default="1"))
