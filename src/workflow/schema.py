from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class SQLResponse(BaseModel):
    """
    Structured response from the LLM for SQL queries.
    """
    table_data: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="List of rows from the database. Each row is a dictionary."
    )
    summary: Optional[str] = Field(
        default=None, 
        description="A natural language explanation or context-aware answer."
    )
