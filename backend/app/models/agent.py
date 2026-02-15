from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import AgentType

class Agent(SQLModel, table=True):
    __tablename__ = "agents"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    agent_code: str
    agent_type: AgentType = Field(default=AgentType.human, sa_type=SAEnum(AgentType, name="agent_type_enum", create_constraint=False, native_enum=True))
    department: Optional[str] = None
    is_active: bool = Field(default=True)
