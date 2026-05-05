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
