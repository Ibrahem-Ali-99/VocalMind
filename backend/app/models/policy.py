from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum
from app.models.enums import ViolationSeverity

class CompanyPolicy(SQLModel, table=True):
    __tablename__ = "company_policies"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    policy_code: str
    category: Optional[str] = None
    policy_text: str
    pinecone_id: Optional[str] = None

class PolicyCompliance(SQLModel, table=True):
    __tablename__ = "policy_compliance"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    policy_id: UUID = Field(foreign_key="company_policies.id")
    is_compliant: Optional[bool] = None
    compliance_score: Optional[float] = None
    violation_severity: Optional[ViolationSeverity] = Field(default=None, sa_type=SAEnum(ViolationSeverity, name="violation_severity_enum", create_constraint=False, native_enum=True))
    confidence_score: Optional[float] = None
    analyzed_by_model: Optional[str] = None
    trigger_description: Optional[str] = None
    evidence_text: Optional[str] = None
    llm_reasoning: Optional[str] = None
    is_human_verified: bool = Field(default=False)
    human_feedback_text: Optional[str] = None
