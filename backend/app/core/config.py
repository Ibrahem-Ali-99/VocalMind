import warnings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "VocalMind API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    FRONTEND_URL: str = "http://localhost:3000"

    # Security
    SECRET_KEY: str = "CHANGE_THIS_TO_A_STRONG_SECRET_KEY"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    # Database (Docker Postgres)
    DATABASE_URL: str = "postgresql+asyncpg://vocalmind:vocalmind_dev@localhost:5432/vocalmind"

    # Emotion API (Docker Container)
    EMOTION_API_URL: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")


settings = Settings()

if settings.SECRET_KEY == "CHANGE_THIS_TO_A_STRONG_SECRET_KEY":
    warnings.warn(
        "SECRET_KEY is using the default value! Set a strong secret via .env "
        "(openssl rand -hex 32). This is insecure for production.",
        stacklevel=1,
    )
