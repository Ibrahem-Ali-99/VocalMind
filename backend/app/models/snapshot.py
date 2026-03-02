from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime, date, timezone
from sqlalchemy import Enum as SAEnum
from app.models.enums import PeriodType


class AgentPerformanceSnapshot(SQLModel, table=True):
    """Pre-aggregated KPI cache per agent per time period."""
    __tablename__ = "agent_performance_snapshots"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    agent_id: UUID = Field(foreign_key="users.id")
    period_type: PeriodType = Field(
        sa_type=SAEnum(PeriodType, name="period_type_enum", create_constraint=False, native_enum=True),
    )
    period_start: date
    period_end: date
    total_interactions: int = Field(default=0)
    avg_overall_score: Optional[float] = None
    avg_empathy_score: Optional[float] = None
    avg_policy_score: Optional[float] = None
    avg_resolution_score: Optional[float] = None
    resolution_rate: Optional[float] = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
