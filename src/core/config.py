import os

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment validation."""

    # LLM Settings
    model_name: str = Field(default="openai/gpt-oss-20b")
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # API Keys
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    groq_api_keys: List[str] = Field(default_factory=list)
    google_api_key: str # Used for Embeddings

    # Database Settings
    database_url: str
    
    # Embedding Settings
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dimensions: int = 1536
    
    # App Settings
    default_user_id: str = "user_123"
    memory_tag: str = "[MEMORIZE]"
    rate_limit_rpm: int = 27
    rate_limit_tpm: int = 2000000
    context_window_size: int = 20
    context_token_limit: int = 8000
    schema_retrieval_limit: int = 6

    @property
    def groq_keys(self) -> List[str]:
        """Returns a list of all available Groq API keys."""
        keys = []
        
        # 1. Check for keys in any env var starting with GROQ_API_KEY
        for env_key, env_val in os.environ.items():
            if env_key.startswith("GROQ_API_KEY"):
                if env_val:
                    if "," in env_val:
                        keys.extend([s.strip() for s in env_val.split(",") if s.strip()])
                    else:
                        keys.append(env_val.strip())
        
        # 2. Check plural list field (explicitly set in Pydantic)
        if self.groq_api_keys:
            for k in self.groq_api_keys:
                if "," in k:
                    keys.extend([s.strip() for s in k.split(",") if s.strip()])
                else:
                    keys.append(k.strip())
        
        # Unique keys only, preserving order
        seen = set()
        unique_keys = [x for x in keys if not (x in seen or seen.add(x))]
        return unique_keys


# Global settings instance
settings = Settings()
