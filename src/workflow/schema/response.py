# ### --- IMPORTS --- ###
from pydantic import BaseModel, Field, ConfigDict

from src.workflow.schema.base import BaseNodeOutput

# ##############################################################################
# [Elaborative Breakdown] Structured Response Models for Multi-Agent Chatbot Interfaces
# Why these schemas?
# After SQL query completion or general intent routing (safety denial/clarification),
# the final response must be formatted with structured schemas to ensure client-side
# components can parse and present the content correctly to the user.
#
# Schemas:
# 1. `ChatbotResponse`: Handles conversational responses for safety denials, general help,
#    or chit-chat, mapping cleanly under the global BaseNodeOutput observability contract.
# 2. `SQLResponse`: Encapsulates the natural language summary of raw database query results,
#    enforcing clean summaries without raw column dumps or unparsed JSON objects.
# ##############################################################################


# ### --- GENERAL RESPONSE SCHEMAS --- ###

class ChatbotResponse(BaseNodeOutput):
    """
    Structured payload for standard conversational responses.
    """
    response: str = Field(
        ..., 
        description="The natural language conversational response or security message."
    )


# ### --- SQL DATABASE RESPONSE SCHEMAS --- ###

class SQLResponse(BaseModel):
    """
    Structured container for summarizing raw SQL execution database outputs.
    """
    model_config = ConfigDict(extra="forbid")
    
    summary: str = Field(
        ..., 
        description="Concise context-aware natural language summary of the returned database records."
    )

