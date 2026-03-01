from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import UserRole, AgentType


class User(SQLModel, table=True):
    """Merged from old users + agents tables.
    agent_type is NULL for admin/manager roles.
    """
    __tablename__ = "users"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    email: str = Field(max_length=320, unique=True, index=True)
    password_hash: str
    name: str = Field(max_length=255)
    role: UserRole = Field(
        sa_type=SAEnum(UserRole, name="user_role_enum", create_constraint=False, native_enum=True),
    )
    is_active: bool = Field(default=True)
    last_login_at: Optional[datetime] = None
    agent_type: Optional[AgentType] = Field(
        default=None,
        sa_type=SAEnum(AgentType, name="agent_type_enum", create_constraint=False, native_enum=True),
    )
