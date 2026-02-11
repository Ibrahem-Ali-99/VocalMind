from typing import AsyncGenerator
from typing import Annotated
from fastapi import Depends
from app.core.database import get_session
from sqlmodel.ext.asyncio.session import AsyncSession

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_db)]
