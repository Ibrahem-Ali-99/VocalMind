from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
import os

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=os.getenv("SQLALCHEMY_ECHO", "").lower() in ("1", "true", "yes"),
    future=True,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "15")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    connect_args={
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0,
    },
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


async def create_db_and_tables():
    from sqlmodel import SQLModel
    # Import ALL models to ensure they are registered with SQLModel.metadata
    from app.models import (  # noqa: F401
        Organization, User, Interaction, Transcript,
        Utterance, EmotionEvent, InteractionScore,
        CompanyPolicy, OrganizationPolicy, PolicyCompliance,
        EmotionFeedback, ComplianceFeedback, InteractionLLMTriggerCache,
    )

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
