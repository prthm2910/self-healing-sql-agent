# ### --- IMPORTS --- ###
from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput

# ##############################################################################
# [Elaborative Breakdown] Structured Intent Security Verdict Schemas
# Why these schemas?
# The Guardian Node stands at the gateway of the entire compilation graph. It acts as
# an immediate gatekeeper checking incoming statements for SQL viability and safety
# parameters.
#
# Components:
# 1. `GuardianOutput`: Establishes the categorized intent enum-equivalent string (SQL,
#    DENY, or CLARIFY) alongside required observability logs.
# 2. `ClarificationOutput`: If categorized as CLARIFY, guides the follow-up generator
#    to output a concise, targeted customer question to break ambiguous query blocks.
# ##############################################################################


# ### --- GUARDIAN SCHEMAS SECTION --- ###

class GuardianOutput(BaseNodeOutput):
    """
    Structured output returned by the gatekeeper Guardian node.
    """
    intent: str = Field(
        ..., 
        description="The categorized intent: SQL (valid domain question), DENY (safety block/chitchat), or CLARIFY."
    )


class ClarificationOutput(BaseModel):
    """
    Structured output container for generating conversational follow-up clarifications.
    """
    model_config = ConfigDict(extra="forbid")
    
    clarification_question: str = Field(
        ..., 
        description="Concise, context-aware follow-up question prompt to clarify user's request details."
    )

