import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment validation."""

    # LLM Settings
    model_name: str = Field(default="meta/llama-3.1-8b-instruct")
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # API Keys
    nvidia_api_key: str
    google_api_key: str
    
    # Database Settings
    database_url: str
    
    # Embedding Settings
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dimensions: int = 1536
    
    # App Settings
    default_user_id: str = "user_123"
    memory_tag: str = "[MEMORIZE]"
    rate_limit_rpm: int = 30
    context_window_size: int = 20

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Global settings instance
settings = Settings()
