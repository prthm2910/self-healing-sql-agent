# ### --- IMPORTS --- ###
import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages

# ##############################################################################
# [Elaborative Breakdown] LangGraph State Reducers & State Management
# Why TypedDict & Annotated Reducers?
# LangGraph coordinates complex, stateful multi-agent workflows by passing an immutable
# state object through a series of asynchronous execution nodes. By defining the state as
# a `TypedDict`, we establish a strict structural contract for what fields are readable
# and writable by any agent in the pipeline.
#
# State Merging Mechanics (Reducers):
# By default, when a node returns a dictionary, LangGraph completely overwrites the
# corresponding keys in the global state. To allow incremental updates, parallel map-reduce
# branches, or append-only histories, we employ Reducer functions via `Annotated[Type, reducer]`:
#
# 1. `add_messages`: Prepend/Append channel that manages the conversation thread. It
#    automatically merges new user or assistant messages into the existing message stream, 
#    ensuring sliding window contexts function correctly.
# 2. `operator.ior` (In-Place Bitwise OR): Used for merging dictionaries.
#    `sql_snippets: Annotated[Dict[str, str], operator.ior]` enables parallel or concurrent
#    Worker Nodes to write their local SQL snippet results (e.g. `{"task_1": "SELECT..."}`)
#    independently. LangGraph safely merges these dictionaries on step resolution instead
#    of overwriting them.
# 3. `operator.add`: Appends elements to lists. Used in `agent_logs` to maintain an audit
#    trail of thought processes across different node invocations without losing earlier history.
#
# Trade-offs:
# - Mutability: Pydantic models are sometimes preferred for strict runtime validation, 
#   but TypedDict has native compatibility with LangGraph's compiler, minimizing serialization
#   overhead between state transitions.
# ##############################################################################


# ### --- STATE DEFINITION SECTION --- ###

class State(TypedDict):
    """
    Unified global execution state for the DVD Rental Self-Healing AI Assistant.
    
    Contains context, structural query metadata, divide-and-conquer steps, 
    and persistent history.
    """
    
    # --- CONVERSATION AND IDENTITY ---
    messages: Annotated[list, add_messages]
    """Active conversational message history thread managed by add_messages reducer."""
    
    user_id: str
    """Unique identifier for the session owner."""
    
    intent: str
    """Guardian validation verdict (SQL, DENY, or CLARIFY)."""
    
    # --- SCHEMA DISCOVERY & ANCHOR SELECTION ---
    is_complex: bool
    """Flag set by Classifier to determine routing down simple or complex divide-and-conquer paths."""
    
    selected_tables: List[str]
    """Array of physical tables identified as relevant during discovery."""
    
    selected_columns: Dict[str, List[str]]
    """Mapping of pruned physical tables to their subset of selected columns."""
    
    fk_relationships: List[Dict[str, str]]
    """Explicit database foreign key relationships linking the selected tables."""

    # --- DIVIDE AND CONQUER WORKFLOW ---
    sub_tasks: List[Dict[str, Any]]
    """Array of decomposed atomic SQL generation tasks for worker execution."""
    
    current_task: Optional[Dict[str, Any]]
    """Active task context passed into a worker node (e.g., in Send loops)."""
    
    join_plan: Dict[str, Any]
    """Deterministic blueprint for joining the worker generated snippets."""
    
    sql_snippets: Annotated[Dict[str, str], operator.ior]
    """Dictionary of generated SQL query snippets mapped to sub-task IDs, merged via in-place OR."""
    
    current_task_index: int
    """Index counter for orchestrating sequential operations and loops."""
    
    agent_logs: Annotated[List[Dict[str, Any]], operator.add]
    """Append-only running execution logs representing the internal reasoning traces."""

    # --- SQL AGENT & EXECUTION ENGINE ---
    current_sql: str
    """The fully assembled or generated SQL statement to execute."""
    
    sql_error: str
    """Standard database error string from execution failures (empty if successful)."""
    
    sql_results: list
    """Raw record list returned by database execution (empty if failure occurred)."""
    
    is_aggregated: bool
    """Flag indicating if SQL output returned a single 1x1 scalar value (e.g. COUNT(*))."""
    
    retry_count: int
    """Counter tracking consecutive self-healing debugging attempts (max 3)."""
    
    # --- CLARIFICATION AND STATE LOCKS ---
    is_awaiting_clarification: bool
    """Lock flag set when user query is too vague, halting execution to ask for details."""
    
    vague_query_context: str
    """Preserved context of the original ambiguous request to anchor follow-up classification."""

