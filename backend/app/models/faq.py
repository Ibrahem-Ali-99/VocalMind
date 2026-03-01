from sqlmodel import SQLModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone


class FAQArticle(SQLModel, table=True):
    """Q&A pairs for the agent FAQ RAG system."""
    __tablename__ = "faq_articles"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    question: str
    answer: str
    category: str = Field(max_length=100)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrganizationFAQArticle(SQLModel, table=True):
    """Junction table: which FAQ articles each organization has activated."""
    __tablename__ = "organization_faq_articles"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_id: UUID = Field(foreign_key="organizations.id")
    article_id: UUID = Field(foreign_key="faq_articles.id")
    is_active: bool = Field(default=True)
    assigned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
