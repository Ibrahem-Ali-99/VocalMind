from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


class InteractionLLMTriggerCache(SQLModel, table=True):
    __tablename__ = "interaction_llm_trigger_cache"
    __table_args__ = (UniqueConstraint("interaction_id", name="uq_interaction_llm_trigger_cache_interaction"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id", index=True)
    org_filter: str | None = Field(default=None, max_length=120)
    report_payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
