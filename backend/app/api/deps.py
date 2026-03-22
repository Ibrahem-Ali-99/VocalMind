from typing import AsyncGenerator
from typing import Annotated
import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlmodel import select
from app.core.database import get_session
from app.core.config import settings
from app.core import security
from app.models.user import User
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

def get_token(request: Request) -> str:
    token = request.cookies.get("vocalmind_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return token

TokenDep = Annotated[str, Depends(get_token)]

async def get_current_user(session: SessionDep, token: TokenDep) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )
    except (jwt.InvalidTokenError, Exception):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    
    statement = select(User).where(User.id == user_id)
    result = await session.exec(statement)
    user = result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]
