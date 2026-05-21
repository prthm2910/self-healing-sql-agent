from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput


class GuardianOutput(BaseNodeOutput):
    """
    Structured output for the Guardian node.
    """
    intent: str = Field(..., description="The categorized intent: SQL, DENY, or CLARIFY.")

class ClarificationOutput(BaseModel):
    """
    Structured output for generating clarification questions.
    """
    model_config = ConfigDict(extra="forbid")
    clarification_question: str = Field(..., description="The concise follow-up question for the user.")
