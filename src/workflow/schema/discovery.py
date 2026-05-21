from typing import List

from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput


class ClassifierOutput(BaseNodeOutput):
    """
    Structured output for the SQL Complexity Classifier.
    """
    is_complex: bool = Field(..., description="True if the query requires joins, complex logic, or multi-table lookups.")

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

class SchemaSelectorOutput(BaseNodeOutput):
    """
    Structured output for the Hybrid Schema Selector.
    """
    selected_tables: List[str] = Field(..., description="Tables identified as relevant.")
    selected_columns: List[ColumnSelection] = Field(..., description="List of table column mappings.")
    fk_relationships: List[FKRelationship] = Field(..., description="Explicit foreign key connections.")
    fk_path_identified: str = Field(..., description="Natural language description of the identified join path.")

class AnchorSelection(BaseModel):
    """
    Structured internal output for identifying anchor tables.
    """
    model_config = ConfigDict(extra="forbid")
    anchors: List[str] = Field(..., description="List of the 2-3 most relevant anchor tables.")
    thought_process: str = Field(..., description="Internal reasoning for selection.")
