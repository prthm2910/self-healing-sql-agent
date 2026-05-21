# ### --- IMPORTS --- ###
from pydantic import BaseModel, ConfigDict, Field, AliasChoices

# ##############################################################################
# [Elaborative Breakdown] Unified Observability contract & Dynamic Aliasing
# Why BaseNodeOutput?
# To achieve absolute deterministic logging, auditability, and observability across all
# agents in the graph, we enforce that every node output inherits from a single base class
# `BaseNodeOutput`. This enforces that every LLM response contains both structured 
# parameters and standard descriptive thought process explanations.
#
# Technical Details & Aliasing:
# 1. `extra="forbid"`: Prevents the LLM from outputting extraneous fields that would waste
#    parsing resources and context tokens, helping with Pydantic V2 parsing compliance.
# 2. `AliasChoices`: LLMs are prone to minor vocabulary deviations (e.g. outputting "reason"
#    or "explanation" instead of "thought_process"). By setting a validation alias with
#    `AliasChoices`, Pydantic gracefully maps these vocabulary shifts into the primary
#    `thought_process` parameter during validation, eliminating parsing errors.
# ##############################################################################


# ### --- BASE SCHEMAS SECTION --- ###

class BaseNodeOutput(BaseModel):
    """
    Base output schema ensuring a strict contract for thought processing and node identity.
    
    Guarantees that every multi-agent node yields descriptive reasoning steps.
    """
    
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    
    node_name: str = Field(
        default="unknown",
        description="The physical name of the node generating the output."
    )
    
    thought_process: str = Field(
        ..., 
        validation_alias=AliasChoices("thought_process", "reason", "explanation"),
        description="A strictly concise summary of your reasoning (max 20 words)."
    )

