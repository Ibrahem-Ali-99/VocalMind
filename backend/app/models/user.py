from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import UserRole

class User(SQLModel, table=True):
    __tablename__ = "users"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    email: str = Field(unique=True, index=True)
    password_hash: Optional[str] = None
    name: str
    role: UserRole = Field(default=UserRole.manager, sa_type=SAEnum(UserRole, name="user_role_enum", create_constraint=False, native_enum=True))
    is_active: bool = Field(default=True)
    last_login_at: Optional[datetime] = None

    # Optional fields from previous version, not in schema but maybe useful for app logic?
    # Keeping them off for now to strictly match schema as requested.
    # picture: Optional[str] = None 
