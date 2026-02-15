from sqlmodel import SQLModel, Field
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import OrgStatus

class Organization(SQLModel, table=True):
    __tablename__ = "organizations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    status: OrgStatus = Field(default=OrgStatus.active, sa_type=SAEnum(OrgStatus, name="org_status_enum", create_constraint=False, native_enum=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
