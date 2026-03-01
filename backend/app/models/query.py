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
    query_mode: QueryMode = Field(
        sa_type=SAEnum(QueryMode, name="query_mode_enum", create_constraint=False, native_enum=True),
    )
    ai_understanding: Optional[str] = None  # renamed from ai_query_understanding
    generated_sql: Optional[str] = None  # renamed from sql_code
    response_text: Optional[str] = None
    execution_time_ms: Optional[int] = None  # new field
