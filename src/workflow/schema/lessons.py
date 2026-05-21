# ### --- IMPORTS --- ###
from typing import List
from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput

# ##############################################################################
# [Elaborative Breakdown] Structured Schemas for Golden Lesson Memory Compilation
# Why these schemas?
# Memory in LLM systems is highly fragile when stored as raw conversation histories.
# To achieve self-improving operational execution, we compile database healing actions
# into structured, semantic objects (Lessons) that act like professional engineering documentation.
#
# Schemas:
# 1. `SQLExample`: Holds comparative query data, explicitly containing the raw query that
#    failed alongside the exact working, debugged fix to support in-context comparison.
# 2. `LessonBody`: Houses the detailed breakdown of the lesson (actionable instruction, 
#    mistake description, and detailed Root Cause Analysis reasoning).
# 3. `LessonDistillationOutput`: The final structured output of the distiller node, 
#    specifying tags for similarity search, application boundaries (global vs table-specific), 
#    and clear developer documentation headers.
# ##############################################################################


# ### --- SQL COMPARISON SCHEMAS --- ###

class SQLExample(BaseModel):
    """
    Comparison container pairing an execution failure query with its validated fix.
    """
    model_config = ConfigDict(extra="forbid")
    
    original_error: str = Field(
        ..., 
        description="The exact raw SQL query string that failed to execute."
    )
    fixed_sql: str = Field(
        ..., 
        description="The corrected, validated SQL query string that executed successfully."
    )


# ### --- CORE LESSON MATERIAL SCHEMAS --- ###

class LessonBody(BaseModel):
    """
    Detailed root cause analysis and dynamic instruction set written by the distiller.
    """
    model_config = ConfigDict(extra="forbid")
    
    instruction: str = Field(
        ..., 
        description="The single, highly actionable golden instruction rule for future agents."
    )
    mistake: str = Field(
        ..., 
        description="A concise description of the exact structural error or hallucination made."
    )
    reasoning: str = Field(
        ..., 
        description="Markdown Root Cause Analysis and systemic prevention explanation."
    )
    example: SQLExample = Field(
        ..., 
        description="Pairwise SQL query comparison showing the bug and correct usage."
    )


# ### --- DISTILLER OUTPUT SCHEMAS --- ###

class LessonDistillationOutput(BaseNodeOutput):
    """
    Structured lesson payload compiled by the Lesson Distiller node.
    """
    is_global: bool = Field(
        ..., 
        description="True if this instruction applies globally to all queries; False if table-specific."
    )
    tags: List[str] = Field(
        ..., 
        description="Dynamic semantic tags enabling targeted filtering in our long-term memory store."
    )
    title: str = Field(
        ..., 
        description="Descriptive, high-impact title representing the core lesson."
    )
    body: LessonBody = Field(
        ..., 
        description="Core lesson material containing actionable instructions and RCAs."
    )
    ending_note: str = Field(
        ..., 
        description="Professional developer sign-off validating systemic adherence."
    )

