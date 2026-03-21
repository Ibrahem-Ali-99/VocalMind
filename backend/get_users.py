import asyncio
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.database import engine
from app.models.user import User

async def get_user():
    async with AsyncSession(engine) as s:
        r = await s.exec(select(User).limit(5))
        users = r.all()
        for u in users:
            print(f"Email: {u.email} | Hash: {u.password_hash}")

if __name__ == "__main__":
    asyncio.run(get_user())
