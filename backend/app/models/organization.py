from sqlmodel import SQLModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlalchemy import Column, Enum as SAEnum
from app.models.enums import OrgStatus


class Organization(SQLModel, table=True):
    __tablename__ = "organizations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(max_length=255)
    status: OrgStatus = Field(
        default=OrgStatus.active,
        sa_type=SAEnum(OrgStatus, name="org_status_enum", create_constraint=False, native_enum=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
