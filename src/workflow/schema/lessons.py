from typing import List

from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput


class SQLExample(BaseModel):
    """Deep structure for code comparisons."""
    model_config = ConfigDict(extra="forbid")
    original_error: str = Field(..., description="The exact SQL query that failed.")
    fixed_sql: str = Field(..., description="The corrected, working SQL query.")

class LessonBody(BaseModel):
    """The core content of the Staff Engineer lesson."""
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(..., description="The single, actionable rule for future agents.")
    mistake: str = Field(..., description="A clear description of the specific error made.")
    reasoning: str = Field(..., description="Markdown Root Cause Analysis and Future Proofing.")
    example: SQLExample = Field(..., description="SQL Query Comparison example.")

class LessonDistillationOutput(BaseNodeOutput):
    """
    Structured output for the Lesson Distiller (Self-Healing).
    """
    is_global: bool = Field(..., description="True if the lesson applies to all queries, False if table-specific.")
    tags: List[str] = Field(..., description="Tags for the schema specific lessons.")
    title: str = Field(..., description="Short, descriptive title for the lesson.")
    body: LessonBody = Field(..., description="The core content of the lesson.")
    ending_note: str = Field(..., description="Professional sign-off (By following this instruction...)")
