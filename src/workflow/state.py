import operator
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
    join_plan: Dict[str, Any] # Structured instructions for the Assembler
    sql_snippets: Annotated[Dict[str, str], operator.ior] # Merged by task_id (ior : In Place OR)
    current_task_index: int # Iterator for sequential or verification loops
    
    agent_logs: Annotated[List[Dict[str, Any]], operator.add] # Append-only log

    # SQL Agent fields
    current_sql: str

    sql_error: str
    sql_results: list
    sql_blueprint: Optional[Dict[str, Any]] # Structured logical tree
    is_aggregated: bool # True if results are a single value (1x1)
    retry_count: int
    # Clarification Lock fields
    is_awaiting_clarification: bool
    vague_query_context: str
