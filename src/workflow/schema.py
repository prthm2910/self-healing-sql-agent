from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class BaseNodeOutput(BaseModel):
    """
    Base class for all node outputs to ensure deterministic observability.
    """
    node_name: str = Field(..., description="Name of the node that produced this output.")
    thought_process: str = Field(..., description="The internal reasoning steps of the agent.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata (latency, tokens, etc.)")

class GuardianOutput(BaseNodeOutput):
    """
    Structured output for the Guardian node.
    """
    intent: str = Field(..., description="The categorized intent: SQL, DENY, or CLARIFY.")

class ClassifierOutput(BaseNodeOutput):
    """
    Structured output for the SQL Complexity Classifier.
    """
    is_complex: bool = Field(..., description="True if the query requires joins, complex logic, or multi-table lookups.")

class SchemaSelectorOutput(BaseNodeOutput):
    """
    Structured output for the Hybrid Schema Selector.
    """
    selected_tables: List[str] = Field(default_factory=list, description="Tables identified as relevant.")
    selected_columns: Dict[str, List[str]] = Field(default_factory=dict, description="Mapping of table names to relevant columns.")
    fk_path_identified: str = Field("", description="Natural language description of the identified join path.")

class AnchorSelection(BaseModel):
    """
    Structured internal output for identifying anchor tables.
    """
    anchors: List[str] = Field(..., description="List of the 2-3 most relevant anchor tables.")
    thought_process: str = Field("", description="Internal reasoning for selection.")

class ClarificationOutput(BaseModel):
    """
    Structured output for generating clarification questions.
    """
    clarification_question: str = Field(..., description="The concise follow-up question for the user.")

class SQLGenerationOutput(BaseModel):
    """
    Structured output for generating or healing SQL.
    """
    sql: str = Field(..., description="The generated or corrected SQL query.")

class ExecuteSQLOutput(BaseModel):
    """
    Structured internal result for SQL execution (non-LLM).
    """
    status: str
    data: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    is_aggregated: bool = False
    error_message: Optional[str] = None

class ChatbotResponse(BaseNodeOutput):
    """
    Structured response for the general chatbot node.
    """
    response: str = Field(..., description="The natural language response from the chatbot.")

class LessonDistillationOutput(BaseNodeOutput):
    """
    Structured output for the Lesson Distiller (Self-Healing).
    """
    is_global: bool = Field(..., description="True if the lesson applies to all queries, False if table-specific.")
    tags: List[str] = Field(default_factory=list, description="Tags for the schema specific lessons.")
    title: str = Field(..., description="Short, descriptive title for the lesson.")
    instruction: str = Field(..., description="The specific rule for future agents to follow.")

class SQLResponse(BaseModel):
    """
    Structured response from the LLM for SQL queries.
    """
    summary: Optional[str] = Field(
        default=None, 
        description="A natural language explanation or context-aware answer."
    )
