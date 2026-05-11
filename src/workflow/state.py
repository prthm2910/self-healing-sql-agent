from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages

class State(TypedDict):
    """The graph state definition."""
    messages: Annotated[list, add_messages]
    user_id: str
    intent: str # Guardian verdict: SQL or DENY
    is_complex: bool # Flag for the Manager to decide routing
    selected_tables: List[str] # Populated by Anchor Selector
    selected_columns: Dict[str, List[str]] # Mapping of table names to relevant columns (Pruned)
    fk_relationships: List[Dict[str, str]] # Explicit foreign key connections between selected tables
    
    # Divide and Conquer fields
    sub_tasks: List[Dict[str, Any]] # List of atomic SQL generation tasks
    join_plan: str # Instructions for the Assembler
    
    agent_logs: List[Dict[str, Any]] # Observability trail for all nodes

    # SQL Agent fields
    current_sql: str

    sql_error: str
    sql_results: list
    is_aggregated: bool # True if results are a single value (1x1)
    retry_count: int
    # Clarification Lock fields
    is_awaiting_clarification: bool
    vague_query_context: str
