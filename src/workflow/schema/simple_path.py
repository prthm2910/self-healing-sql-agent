from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class SQLGenerationOutput(BaseModel):
    """
    Structured output for generating or healing SQL.
    """
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(..., description="The generated or corrected SQL query.")

class ExecuteSQLOutput(BaseModel):
    """
    Structured internal result for SQL execution (non-LLM).
    Used for local state, not LLM output.
    """
    status: str
    data: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    is_aggregated: bool = False
    error_message: Optional[str] = None
