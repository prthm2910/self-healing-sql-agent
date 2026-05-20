from typing import List, Optional, Any, Dict, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

class BaseNodeOutput(BaseModel):
    """
    Base class for all node outputs to ensure deterministic observability.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    node_name: str = "unknown"
    thought_process: str = Field(
        ..., 
        validation_alias=AliasChoices("thought_process", "reason", "explanation"),
        description="A strictly concise summary of your reasoning (max 20 words)."
    )

# --- SQL OBJECT MODELS (The Blueprint System) ---

class SQLFilter(BaseModel):
    """Structured WHERE condition."""
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., description="Column name (with table alias if necessary).")
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "LIKE", "ILIKE", "IN", "IS NULL"]
    value: Any = Field(..., description="The literal value, list of values, or NULL.")

class SQLSelection(BaseModel):
    """Structured SELECT field."""
    model_config = ConfigDict(extra="forbid")
    column: str = Field(..., description="Column name or '*'")
    aggregation: Optional[Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]] = None
    alias: Optional[str] = None

class QueryBlueprint(BaseNodeOutput):
    """The structured 'Logical Tree' for a SQL query."""
    task_id: int = Field(default=1)
    tables: List[str] = Field(..., description="List of tables involved.")
    select: List[SQLSelection] = Field(..., description="Columns/aggregations to select.")
    filters: List[SQLFilter] = Field(default_factory=list, description="List of logical filters.")
    group_by: List[str] = Field(default_factory=list, description="List of columns for grouping.")
    order_by: List[Dict[str, Literal["ASC", "DESC"]]] = Field(default_factory=list)
    limit: Optional[int] = None

class SQLResultSet(BaseModel):
    """Structured envelope for database output."""
    rows: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    status: Literal["success", "error"]
    error_message: Optional[str] = None

# --- GUARDIAN TIER ---

class GuardianLLMOutput(BaseNodeOutput):
    """Fields populated by the LLM in the Guardian node."""
    intent: str = Field(..., description="The categorized intent: SQL, DENY, or CLARIFY.")

class GuardianNodeOutput(GuardianLLMOutput):
    """Final output from the Guardian node for state updates."""
    is_awaiting_clarification: bool = Field(False)
    vague_query_context: str = Field("")
    agent_logs: List[Dict[str, Any]] = Field(default_factory=list)

# --- CLASSIFIER TIER ---

class ClassifierLLMOutput(BaseNodeOutput):
    """Fields populated by the LLM in the Classifier node."""
    is_complex: bool = Field(..., description="True if the query requires joins, complex logic, or multi-table lookups.")

class ClassifierNodeOutput(ClassifierLLMOutput):
    """Final output from the Classifier node for state updates."""
    agent_logs: List[Dict[str, Any]] = Field(default_factory=list)

# --- SCHEMA SELECTOR TIER ---

class ColumnSelection(BaseModel):
    """Mapping of table names to relevant columns."""
    model_config = ConfigDict(extra="forbid")
    table_name: str = Field(..., description="Physical table name.")
    columns: List[str] = Field(..., description="List of column names.")

class FKRelationship(BaseModel):
    """Explicit foreign key connection."""
    model_config = ConfigDict(extra="forbid")
    source_table: str = Field(..., description="Source table.")
    source_column: str = Field(..., description="Source column.")
    target_table: str = Field(..., description="Target table.")
    target_column: str = Field(..., description="Target column.")

class SchemaSelectorLLMOutput(BaseNodeOutput):
    """Fields populated by the LLM in the Schema Selector node."""
    selected_columns: List[ColumnSelection] = Field(..., description="List of table column mappings.")
    fk_relationships: List[FKRelationship] = Field(default_factory=list, description="Explicit foreign key connections.")

class SchemaSelectorNodeOutput(SchemaSelectorLLMOutput):
    """Final output from the Schema Selector node for state updates."""
    selected_tables: List[str] = Field(default_factory=list)
    fk_path_identified: str = Field("")
    agent_logs: List[Dict[str, Any]] = Field(default_factory=list)

# --- DIVIDE & CONQUER (MANAGER/WORKER) TIER ---

class SubTask(BaseModel):
    """An atomic unit of work for a Worker node."""
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(..., description="Unique ID for the sub-task (e.g., 'task_1').")
    description: str = Field(..., description="Description of what this sub-task generates.")
    tables: List[str] = Field(..., description="Tables involved in this specific snippet.")
    required_columns: List[str] = Field(..., description="Columns that MUST be in the SELECT list for joins.")
    dependencies: List[str] = Field(..., description="IDs of tasks this task depends on.")

class JoinStep(BaseModel):
    """A single join operation between two sub-task snippets."""
    model_config = ConfigDict(extra="forbid")
    left: str = Field(..., description="Task ID of the left side (or 'base').")
    right: str = Field(..., description="Task ID of the right side snippet.")
    on: str = Field(..., description="The column name to join on (assumed same in both).")
    join_type: str = Field(..., description="inner, left, or cross.")

class JoinPlan(BaseModel):
    """The blueprint for assembling multiple SQL snippets."""
    model_config = ConfigDict(extra="forbid")
    base_task: str = Field(..., description="The ID of the primary task to start the FROM clause.")
    steps: List[JoinStep] = Field(..., description="Ordered steps to join additional snippets.")
    final_select: str = Field(..., description="The final columns to select from the joined set.")

class DecomposerOutput(BaseNodeOutput):
    """The structured plan generated by the Manager for Divide and Conquer."""
    sub_tasks: List[SubTask] = Field(..., description="List of atomic SQL generation tasks.")
    join_plan: JoinPlan = Field(..., description="Deterministic blueprint for the SQLTranspiler.")
    complexity_score: int = Field(..., ge=1, le=10, description="Estimated complexity (1-10).")

# --- SELF-HEALING & LESSONS ---

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
    """Structured output for the Lesson Distiller."""
    is_global: bool = Field(..., description="True if the lesson applies to all queries, False if table-specific.")
    tags: List[str] = Field(default_factory=list, description="Tags for the schema specific lessons.")
    title: str = Field(..., description="Short, descriptive title for the lesson.")
    body: LessonBody = Field(..., description="The core content of the lesson.")

# --- RESPONSE RENDERER ---

class SQLResponse(BaseNodeOutput):
    """Structured response from the LLM for SQL queries."""
    summary: Optional[str] = Field(default=None, description="A natural language explanation or context-aware answer.")
