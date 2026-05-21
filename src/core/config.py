import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment validation."""

    # LLM Settings
    model_name: str = Field(default="openai/gpt-oss-20b")
    
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Database Settings
    database_url: str = os.getenv("DATABASE_URL", "")
    
    # Embedding Settings
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dimensions: int = 1536
    
    # App Settings
    default_user_id: str = "user_123"
    memory_tag: str = "[MEMORIZE]"
    rate_limit_rpm: int = 25
    context_window_size: int = 20

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Global settings instance
settings = Settings()
