# ### --- IMPORTS --- ###
from typing import List
from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput

# ##############################################################################
# [Elaborative Breakdown] Structured Relational Metadata Schema Discovery Models
# Why structured schemas?
# Unstructured LLM outputs are highly prone to formatting drift, syntax errors, and
# incomplete JSON records. We define a precise suite of Pydantic models for the
# Discovery node outputs.
#
# Key Components:
# 1. `ClassifierOutput`: Enforces boolean classification (`is_complex`) to drive 
#    branching decisions in the compiler graph.
# 2. `ColumnSelection` & `FKRelationship`: Ensures a strictly validated shape for mapping
#    selected columns and concrete database foreign keys.
# 3. `SchemaSelectorOutput`: Holds the pruned schema blueprint, consolidating table selections,
#    individual columns, foreign key connections, and join-paths into a validated model 
#    before passing it downstream to the generator nodes.
# 4. `AnchorSelection`: Intermediate anchor selectors representing extracted high-priority 
#    tables to start search queries on.
# ##############################################################################


# ### --- COMPLEXITY CLASSIFIER SCHEMAS --- ###

class ClassifierOutput(BaseNodeOutput):
    """
    Structured output payload for the SQL Complexity Intent Classifier.
    """
    is_complex: bool = Field(
        ..., 
        description="True if the query requires joins, complex logic, or multi-table lookups."
    )


# ### --- RELATIONAL METADATA SCHEMAS --- ###

class ColumnSelection(BaseModel):
    """
    Structured container mapping a single table to its subset of relevant physical columns.
    """
    model_config = ConfigDict(extra="forbid")
    
    table_name: str = Field(
        ..., 
        description="Target physical table name in the relational database."
    )
    columns: List[str] = Field(
        ..., 
        description="List of specific columns selected for projection or filtering."
    )


class FKRelationship(BaseModel):
    """
    Structured model defining a single foreign key connection between two database tables.
    """
    model_config = ConfigDict(extra="forbid")
    
    source_table: str = Field(
        ..., 
        description="Physical table containing the foreign key column."
    )
    source_column: str = Field(
        ..., 
        description="Foreign key column name linking to the target table key."
    )
    target_table: str = Field(
        ..., 
        description="Referenced primary key table."
    )
    target_column: str = Field(
        ..., 
        description="Primary key column name referenced in the target table."
    )


# ### --- SCHEMA SELECTION SCHEMAS --- ###

class SchemaSelectorOutput(BaseNodeOutput):
    """
    Consolidated structural schema payload representing a pruned query diagram.
    """
    selected_tables: List[str] = Field(
        ..., 
        description="List of physical tables identified as relevant for the database operation."
    )
    selected_columns: List[ColumnSelection] = Field(
        ..., 
        description="List of selected table-column projection mappings."
    )
    fk_relationships: List[FKRelationship] = Field(
        ..., 
        description="Array of explicitly mapped foreign key join relationships linking selected tables."
    )
    fk_path_identified: str = Field(
        ..., 
        description="Natural language summary of the discovered join path traversal."
    )


class AnchorSelection(BaseModel):
    """
    Structured schema representing semantically discovered anchor tables in Pagila.
    """
    model_config = ConfigDict(extra="forbid")
    
    anchors: List[str] = Field(
        ..., 
        description="List of the 2-3 most semantically relevant anchor tables."
    )
    thought_process: str = Field(
        ..., 
        description="Detailed internal reasoning validating the selection."
    )

