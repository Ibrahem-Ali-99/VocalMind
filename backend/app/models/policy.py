from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID, uuid4


class CompanyPolicy(SQLModel, table=True):
    """Global policy definitions. Text is chunked and embedded into Pinecone."""
    __tablename__ = "company_policies"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    policy_category: str = Field(max_length=100)
    policy_title: str = Field(max_length=255)
    policy_text: str
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrganizationPolicy(SQLModel, table=True):
    """Junction table: which policies each organization has activated."""
    __tablename__ = "organization_policies"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    policy_id: UUID = Field(foreign_key="company_policies.id")
    is_active: bool = Field(default=True)
    assigned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PolicyCompliance(SQLModel, table=True):
    """AI compliance verdict for a policy against an interaction."""
    __tablename__ = "policy_compliance"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    interaction_id: UUID = Field(foreign_key="interactions.id")
    policy_id: UUID = Field(foreign_key="company_policies.id")
    is_compliant: bool
    compliance_score: float
    llm_reasoning: Optional[str] = None
    evidence_text: Optional[str] = None
    retrieved_policy_text: Optional[str] = None
