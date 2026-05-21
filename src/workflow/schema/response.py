from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput


class ChatbotResponse(BaseNodeOutput):
    """
    Structured response for the general chatbot node.
    """
    response: str = Field(..., description="The natural language response from the chatbot.")

class SQLResponse(BaseModel):
    """
    Structured response from the LLM for SQL queries.
    """
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(..., description="A natural language explanation or context-aware answer.")
