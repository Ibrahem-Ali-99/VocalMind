from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from datetime import datetime, timezone
from app.models.enums import QueryMode


class AssistantQuery(SQLModel, table=True):
    __tablename__ = "assistant_queries"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id")
    organization_id: UUID = Field(foreign_key="organizations.id")
    query_mode: QueryMode = Field(
        sa_type=SAEnum(QueryMode, name="query_mode_enum", create_constraint=False, native_enum=True),
    )
    audio_input_path: Optional[str] = None
    query_text: str
    ai_understanding: Optional[str] = None
    generated_sql: Optional[str] = None
    response_text: Optional[str] = None
    execution_time_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
