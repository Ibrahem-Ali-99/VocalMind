# Central API aggregator.
# Logic: All domain routers are registered here — one clean import in app/main.py.

from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.emotion import router as emotion_router

api_router = APIRouter()

api_router.include_router(auth_router.router, prefix="/auth", tags=["auth"])
api_router.include_router(emotion_router.router, prefix="/emotion", tags=["emotion"])
