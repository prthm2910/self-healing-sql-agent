# ### --- IMPORTS --- ###
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput

# ##############################################################################
# [Elaborative Breakdown] Structured Schemas for AST Divide and Conquer Query Compilation
# Why these schemas?
# For highly complex questions requiring multi-table joins, standard single-shot LLM
# generation suffers from high regression rates. Our system divides the logic into
# smaller "Islands of Logic" and then recomposes them deterministically via a compiler.
#
# Schemas:
# 1. `SubTask`: Establishes the contract for a single worker task, specifying involved 
#    tables and protecting join keys by forcing them into required SELECT lists.
# 2. `JoinStep`: Defines a single pairwise join operation (left, right, on key, join type)
#    which functions as a directed compile instruction.
# 3. `JoinPlan`: Houses the complete deterministic recipe to stitch the worker snippets
#    together into a unified query.
# 4. `DecomposerOutput`: The manager's roadmap, yielding both the subtasks and the compile
#    join plan.
# 5. `WorkerOutput`: The single worker output holding its generated snippet.
# ##############################################################################


# ### --- ATOMIC SUB-TASK SCHEMAS --- ###

class SubTask(BaseModel):
    """
    Represents an atomic, self-contained unit of SQL generation for a single Worker Node.
    """
    model_config = ConfigDict(extra="forbid")
    
    task_id: str = Field(
        ..., 
        description="Unique ID for the sub-task (e.g., 'task_1')."
    )
    description: str = Field(
        ..., 
        description="Description of what this sub-task generates (the prompt for the worker)."
    )
    tables: List[str] = Field(
        ..., 
        description="Physical database tables involved in this specific snippet."
    )
    required_columns: List[str] = Field(
        ..., 
        description="Columns that MUST be in the SELECT list to enable subsequent dynamic joins."
    )
    dependencies: List[str] = Field(
        ..., 
        description="IDs of tasks this specific sub-task depends on."
    )


# ### --- DETAILED JOIN BLUEPRINT SCHEMAS --- ###

class JoinStep(BaseModel):
    """
    Structured directive instructing the compiler how to join two query snippets.
    """
    model_config = ConfigDict(extra="forbid")
    
    left: str = Field(
        ..., 
        description="Task ID or accumulated alias representing the left-hand operand."
    )
    right: str = Field(
        ..., 
        description="Task ID snippet representing the right-hand operand to join."
    )
    on: str = Field(
        ..., 
        description="The matching column name to execute the join on (assumed identical)."
    )
    join_type: str = Field(
        ..., 
        description="The relational join operator: inner, left, or cross."
    )


class JoinPlan(BaseModel):
    """
    Deterministic plan outlining the full join layout to stitch query snippets.
    """
    model_config = ConfigDict(extra="forbid")
    
    base_task: str = Field(
        ..., 
        description="The primary sub-task ID representing the FROM clause."
    )
    steps: List[JoinStep] = Field(
        ..., 
        description="Ordered sequence of steps to join remaining snippets."
    )
    final_select: str = Field(
        ..., 
        description="Final SELECT projection columns list, strictly using table aliases (alias.column)."
    )
    where: Optional[str] = Field(
        None, 
        description="Optional global WHERE constraints applied post-assembly."
    )
    order_by: Optional[str] = Field(
        None, 
        description="Optional global ordering criteria."
    )
    limit: Optional[int] = Field(
        None, 
        description="Optional maximum limit capping returned records."
    )


# ### --- OUTPUT PAYLOAD SCHEMAS --- ###

class DecomposerOutput(BaseNodeOutput):
    """
    Structured plan returned by the Manager node to coordinate Divide and Conquer.
    """
    sub_tasks: List[SubTask] = Field(
        ..., 
        description="List of isolated atomic SQL worker tasks."
    )
    join_plan: JoinPlan = Field(
        ..., 
        description="Stitching blueprint for query assembly."
    )
    complexity_score: int = Field(
        ..., 
        ge=1, 
        le=10, 
        description="Estimated query plan complexity rating (1-10)."
    )


class WorkerOutput(BaseNodeOutput):
    """
    Structured output returned by an isolated SQL worker node.
    """
    sql: str = Field(
        ..., 
        description="The generated isolated SQL query snippet."
    )

