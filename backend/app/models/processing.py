from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import Enum as SAEnum
from app.models.enums import JobStage, JobStatus


class ProcessingJob(SQLModel, table=True):
    """Tracks each async pipeline stage per interaction independently."""
    __tablename__ = "processing_jobs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    stage: JobStage = Field(
        sa_type=SAEnum(JobStage, name="job_stage_enum", create_constraint=False, native_enum=True),
    )
    status: JobStatus = Field(
        default=JobStatus.pending,
        sa_type=SAEnum(JobStatus, name="job_status_enum", create_constraint=False, native_enum=True),
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)
