# Consolidated schemas for the auth domain.
# Logic: Colocated schemas prevent jumping between schema/models/routes folders.

from typing import Optional
from pydantic import BaseModel
from uuid import UUID


# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[int] = None


# --- User Schemas ---
class UserBase(BaseModel):
    email: Optional[str] = None
    is_active: Optional[bool] = True
    name: Optional[str] = None


class UserCreate(UserBase):
    email: str
    password: Optional[str] = None


class User(UserBase):
    id: Optional[UUID] = None

    class Config:
        from_attributes = True


class GoogleUser(BaseModel):
    email: str
    name: str
    picture: Optional[str] = None
