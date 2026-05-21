# ### --- IMPPORTS --- ###
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ##############################################################################
# [Elaborative Breakdown] Pydantic Settings Validation & Environment Isolation
# Why Pydantic Settings?
# In enterprise web services, using raw environment variables directly throughout the
# application code introduces tight coupling, error-prone string parsing, and lack of
# type safety. By utilizing Pydantic's `BaseSettings`, we declare a typed, central
# contract for all runtime configurations. Pydantic automatically handles parsing, type
# coercion (e.g., converting string environment variables to integers like `rate_limit_rpm`),
# and validation during boot time, causing the application to "fail fast" if a critical
# variable is missing or malformed.
#
# Environment Prioritization:
# SettingsConfigDict defines `env_file=".env"`. This means Pydantic loads values in 
# the following strict hierarchy (highest priority first):
# 1. Environment variables set directly in the shell or OS process.
# 2. Values specified in the local `.env` configuration file.
# 3. Default fallback values specified in the Python model fields.
#
# Trade-offs:
# - Boot Latency: There is a microsecond parsing overhead during Pydantic initialization, 
#   but since it executes exactly once on module load, it has zero impact on active request
#   handling.
# - Security Boundary: API keys are loaded cleanly without hardcoded strings. If no key is
#   found, empty fallbacks are provided to permit offline or test-mock initialization.
# ##############################################################################


# ### --- SETTINGS CLASS --- ###

class Settings(BaseSettings):
    """
    Application settings, configurations, and process-level environment validator.
    
    Provides strict type coercion and parsing from environment variables and .env files.
    """

    # --- LLM CONFIGURATION ---
    model_name: str = Field(
        default="openai/gpt-oss-20b", 
        description="Target Large Language Model identifier string."
    )
    
    # --- CREDENTIALS AND ENDPOINTS ---
    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", ""),
        description="API Authorization credential token for the Groq platform."
    )
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="API Authorization credential token for Google Generative AI."
    )
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", ""),
        description="PostgreSQL Database Connection URI string."
    )
    
    # --- EMBEDDING CONFIGURATION ---
    embedding_model: str = Field(
        default="models/gemini-embedding-2",
        description="Target semantic embedding model identifier."
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality length of generated embedding vectors."
    )
    
    # --- RESOURCE CONTROLS AND MEMORY ---
    default_user_id: str = Field(
        default="user_123",
        description="Default identifier for non-authenticated active session users."
    )
    memory_tag: str = Field(
        default="[MEMORIZE]",
        description="Special trigger tag parsed by the LLM response flow to invoke long-term storage."
    )
    rate_limit_rpm: int = Field(
        default=25,
        description="Maximum allowed requests per minute per user context."
    )
    context_window_size: int = Field(
        default=20,
        description="Maximum historical chat message count to keep in memory."
    )
    token_per_minute: int = Field(
        default=500000,
        description="Global token-per-minute ceiling for LLM usage monitoring."
    )

    # Configuration metadata targeting local env file loading
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ### --- APPLICATION INSTANCE --- ###

# Global settings instance loaded and validated at module import time
settings: Settings = Settings()

