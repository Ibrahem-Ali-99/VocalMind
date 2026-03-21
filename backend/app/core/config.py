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

    # Google OAuth / AI
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"
    GOOGLE_API_KEY: str = ""

    # Database (Docker Postgres)
    DATABASE_URL: str = "postgresql+asyncpg://vocalmind:vocalmind_dev@localhost:5432/vocalmind"

    # AI service routing: True = Docker containers, False = Kaggle server
    IS_LOCAL: bool = True

    # Docker container URLs (used when IS_LOCAL=true)
    EMOTION_API_URL: str = "http://localhost:8001"
    VAD_API_URL: str = "http://localhost:8002"

    # Kaggle Emotion API (ngrok URL)
    KAGGLE_NGROK_URL: str = ""
    KAGGLE_API_SECRET: str = "vocalmind_secret_gpu_key"

    # Kaggle server URL (used when IS_LOCAL=false)
    KAGGLE_SERVER_URL: str = ""

    # Supabase (for routes that use Supabase client directly)
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # OpenAI API Key for Assistant
    OPENAI_API_KEY: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")


settings = Settings()

if settings.SECRET_KEY == "CHANGE_THIS_TO_A_STRONG_SECRET_KEY":
    warnings.warn(
        "SECRET_KEY is using the default value! Set a strong secret via .env "
        "(openssl rand -hex 32). This is insecure for production.",
        stacklevel=1,
    )
