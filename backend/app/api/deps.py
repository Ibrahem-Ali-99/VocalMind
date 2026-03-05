from typing import AsyncGenerator
from typing import Annotated
from fastapi import Depends
from app.core.database import get_session
from app.core.config import settings
from sqlmodel.ext.asyncio.session import AsyncSession
from supabase import create_client, Client


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


def get_supabase() -> Client:
    """Create a Supabase client for routes that use Supabase directly."""
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY,
    )


SessionDep = Annotated[AsyncSession, Depends(get_db)]
