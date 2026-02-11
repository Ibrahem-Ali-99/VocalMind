from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import QueryMode

class ManagerQuery(SQLModel, table=True):
    __tablename__ = "manager_queries"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id")
    organization_id: UUID = Field(foreign_key="organizations.id")
    query_text: str
    query_mode: QueryMode = Field(default=QueryMode.chat, sa_type=SAEnum(QueryMode, name="query_mode_enum", create_constraint=False, native_enum=True))
    ai_query_understanding: Optional[str] = None
    sql_code: Optional[str] = None
    response_text: Optional[str] = None
    retrieved_policy_id: Optional[UUID] = Field(default=None, foreign_key="company_policies.id")
