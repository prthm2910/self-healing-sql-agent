from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages

class State(TypedDict):
    """The graph state definition."""
    messages: Annotated[list, add_messages]
    user_id: str
    intent: str # Guardian verdict: SQL or DENY
    is_complex: bool # Flag for the Manager to decide routing
    selected_tables: List[str] # Populated by Schema Selector
    agent_logs: List[Dict[str, Any]] # Observability trail for all nodes
    
    # SQL Agent fields
    db_schema: Optional[Dict[str, List[str]]]
    current_sql: str
    sql_error: str
    sql_results: list
    is_aggregated: bool # True if results are a single value (1x1)
    retry_count: int
    # Clarification Lock fields
    is_awaiting_clarification: bool
    vague_query_context: str
