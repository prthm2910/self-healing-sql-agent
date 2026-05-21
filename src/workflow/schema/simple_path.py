# ### --- IMPORTS --- ###
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

# ##############################################################################
# [Elaborative Breakdown] Simple Direct Path Query Generation & Execution Schemas
# Why these schemas?
# For standard, non-join, single-table database queries, utilizing complex decomposer
# manager networks is counter-productive. The system routes simple queries through an
# optimized single-step SQL generation node.
#
# Schemas:
# 1. `SQLGenerationOutput`: Direct container validating that the LLM generates a single
#    clean SQL string matching PostgreSQL syntax specs.
# 2. `ExecuteSQLOutput`: A local, internal service schema (non-LLM) validating the database
#    driver execution results. It encapsulates records, query row count, aggregation indicators
#    (e.g., if a scalar COUNT value was project), and error tracebacks to support upstream 
#    self-healing retry logic.
# ##############################################################################


# ### --- SQL GENERATION SCHEMAS --- ###

class SQLGenerationOutput(BaseModel):
    """
    Structured query output returned by direct single-shot SQL generation nodes.
    """
    model_config = ConfigDict(extra="forbid")
    
    sql: str = Field(
        ..., 
        description="The generated or corrected PostgreSQL query string."
    )


# ### --- INTERNAL DB DRIVER EXECUTION SCHEMAS --- ###

class ExecuteSQLOutput(BaseModel):
    """
    Structured payload representing database engine execution results (internal use).
    """
    
    status: str = Field(
        ...,
        description="Operational execution verdict (e.g. 'success' or 'error')."
    )
    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw list of dictionary rows representing retrieved records."
    )
    row_count: int = Field(
        default=0,
        description="Total row count project within the active query data set."
    )
    is_aggregated: bool = Field(
        default=False,
        description="Flag indicating if the query projected a scalar summary value (e.g., COUNT)."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Raw traceback or exception message string if database execution failed."
    )

