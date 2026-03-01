from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=True, future=True)


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
        EmotionFeedback, ComplianceFeedback, ManagerQuery, HumanFeedback,
    )

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
