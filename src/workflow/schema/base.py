from pydantic import BaseModel, ConfigDict, Field, AliasChoices

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
