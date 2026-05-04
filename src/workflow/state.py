from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """The graph state definition."""
    messages: Annotated[list, add_messages]
    user_id: str
    intent: str # Added field to persist Guardian verdict
    
    # SQL Agent fields
    current_sql: str
    sql_error: str
    sql_results: list
    retry_count: int
    previous_failed_query: str
