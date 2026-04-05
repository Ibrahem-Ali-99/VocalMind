from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import create_db_and_tables
from app.core.interaction_processing import start_processing_worker, stop_processing_worker
from app.api.main import api_router
from scripts.seed_nexalink import main as seed_nexalink_main


from app.api.routes.dashboard import prewarm_dashboard_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    await seed_nexalink_main()
    await start_processing_worker()
    # Pre-warm the dashboard cache so the first manager load is instantaneous
    import asyncio
    asyncio.create_task(prewarm_dashboard_cache())
    try:
        yield
    finally:
        await stop_processing_worker()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

# Single router include — all domains registered in app/api/main.py
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {"message": "Welcome to VocalMind API"}


@app.get("/health")
def health():
    return {"status": "ok"}
