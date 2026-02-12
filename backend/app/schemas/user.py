from typing import Optional
from pydantic import BaseModel
from uuid import UUID

# Shared properties
class UserBase(BaseModel):
    email: Optional[str] = None
    is_active: Optional[bool] = True
    name: Optional[str] = None

# Properties to receive via API on creation
class UserCreate(UserBase):
    email: str
    password: Optional[str] = None

# Properties to return via API
class User(UserBase):
    id: Optional[UUID] = None

    class Config:
        from_attributes = True

# Google Token Payload
class GoogleUser(BaseModel):
    email: str
    name: str
    picture: Optional[str] = None
