import asyncio

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.database import engine
from app.models.user import User


async def get_user() -> None:
    async with AsyncSession(engine) as session:
        result = await session.exec(select(User).limit(5))
        for user in result.all():
            print(f"Email: {user.email} | Hash: {user.password_hash}")


if __name__ == "__main__":
    asyncio.run(get_user())
