# Auth router: password login + Google OAuth (direct and redirect flows).
# Logic: All auth HTTP handlers in one file, right next to the service and schemas they use.

import secrets
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import select

from app.api.deps import SessionDep
from app.core import security
from app.core.config import settings
from app.models.organization import Organization
from app.models.user import User as UserModel
from app.api.routes.auth.schemas import Token
from app.api.routes.auth.service import verify_google_token

router = APIRouter()

# In-memory state store (use Redis in production at scale)
_oauth_states: set[str] = set()


async def _get_or_create_user(session: SessionDep, email: str, name: str) -> UserModel:
    """Find existing user by email, or create a new OAuth user."""
    statement = select(UserModel).where(UserModel.email == email)
    result = await session.exec(statement)
    user = result.first()

    if user:
        return user

    org_result = await session.exec(select(Organization))
    org = org_result.first()
    if not org:
        org = Organization(name="Default Organization")
        session.add(org)
        await session.commit()
        await session.refresh(org)

    user = UserModel(
        organization_id=org.id,
        email=email,
        name=name,
        password_hash=None,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


def _create_token(user_id: Any) -> dict:
    expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": security.create_access_token(user_id, expires_delta=expires),
        "token_type": "bearer",
    }


# ---------- Password login ----------

@router.post("/login/access-token", response_model=Token)
async def login_access_token(
    session: SessionDep,
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """OAuth2 compatible token login."""
    statement = select(UserModel).where(UserModel.email == form_data.username)
    result = await session.exec(statement)
    user = result.first()

    if (
        not user
        or not user.password_hash
        or not security.verify_password(form_data.password, user.password_hash)
    ):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return _create_token(user.id)


# ---------- Google OAuth (direct ID-token from frontend) ----------

@router.post("/google", response_model=Token)
async def google_auth(token: str, session: SessionDep) -> Any:
    """Google Login: Verify Google ID token and return access token."""
    google_user = verify_google_token(token)
    if not google_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Google token")

    user = await _get_or_create_user(session, google_user.email, google_user.name)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return _create_token(user.id)


# ---------- Google OAuth (redirect flow) ----------

@router.get("/google/login")
async def login_google():
    """Redirect to Google's consent screen."""
    state = secrets.token_urlsafe(32)
    _oauth_states.add(state)
    params = urlencode({
        "response_type": "code",
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "scope": "openid email profile",
        "access_type": "offline",
        "state": state,
    })
    return RedirectResponse(url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@router.get("/google/callback")
async def google_callback(code: str, state: str, session: SessionDep):
    """Exchange authorization code for tokens and log the user in."""
    if state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    _oauth_states.discard(state)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        token_data = response.json()

    if "error" in token_data:
        raise HTTPException(status_code=400, detail=token_data.get("error_description", "Token exchange failed"))

    google_user = verify_google_token(token_data.get("id_token"))
    if not google_user:
        raise HTTPException(status_code=400, detail="Invalid Google token")

    user = await _get_or_create_user(session, google_user.email, google_user.name)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    token = security.create_access_token(
        user.id, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return RedirectResponse(url=f"{settings.FRONTEND_URL}/login/success?token={token}")
