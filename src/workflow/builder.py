# ### --- IMPORTS --- ###
from typing import Literal, List, Dict, Any, Union, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.graph.state import CompiledStateGraph

from src.workflow.state import State
from src.workflow.nodes.guardian import guardian_node, clarify_node
from src.workflow.nodes.discovery import classifier_node, anchor_selector_node, column_pruner_node
from src.workflow.nodes.complex_path import decomposer_node, worker_node, assembler_node
from src.workflow.nodes.simple_path import generate_sql_node, execute_sql_node, heal_sql_node
from src.workflow.nodes.response import format_sql_response_node
from src.services.database import get_connection_pool
from src.services.embeddings import get_embeddings_provider
from src.core.config import settings

# ##############################################################################
# [Elaborative Breakdown] LangGraph Compilation, Routing, and Parallel Scatter-Gather
# Why StateGraph & Send?
# This module compiles our multi-agent self-healing SQL generation graph. It leverages
# LangGraph's dynamic execution system to orchestrate conditional transitions, long-term 
# checkpointing, and high-performance concurrent processing.
#
# Key Architectural Patterns:
# 1. State-Driven Conditional Routing:
#    Rather than static paths, the graph uses lightweight functional routers 
#    (`guardian_router`, `classifier_router`, `healing_router`) to analyze the current
#    State and dynamically yield the identifier of the next execution node.
# 2. Parallel Scatter-Gather (Map-Reduce) via Send:
#    In the Complex Query path, the `decomposer_router` divides a complex prompt into
#    independent atomic sub-tasks. It returns an array of `Send` objects:
#    `Send("worker", {"current_task": task})`.
#    LangGraph compiles this into parallel isolated workers execution. Each worker
#    spins up independently, generates its local SQL snippet, and writes to the state 
#    channel `sql_snippets`, which automatically combines results via its `operator.ior` 
#    reducer. Once all parallel worker instances terminate, execution "gathers" back to 
#    the `assembler` node.
# 3. Durable Checkpointing & Semantic Memory Store:
#    - `PostgresSaver` acts as our ACID-compliant transaction log, saving state checkpoints
#      after every agent step to allow seamless session resumption and error resilience.
#    - `PostgresStore` acts as our long-term vector semantic memory. It uses our 
#      custom embeddings provider to index experiences (golden lessons) under a
#      vector space in PostgreSQL, permitting immediate similarity lookups across sessions.
# ##############################################################################


# ### --- ROUTERS SECTION --- ###

def guardian_router(state: State) -> Literal["classifier", "clarify", "end"]:
    """
    Evaluates the security guardian's verdict to route the conversation workflow.
    
    Args:
        state: The current global execution State.
        
    Returns:
        The string name of the next node to transition to ("classifier", "clarify", or "end").
    """
    intent: str = state.get("intent", "DENY")
    if intent == "SQL":
        return "classifier"
    if intent == "CLARIFY":
        return "clarify"
    return "end"


def classifier_router(state: State) -> Literal["anchor_selector", "generate_sql"]:
    """
    Routes queries down simple single-table or complex divide-and-conquer paths.
    
    Args:
        state: The current global execution State.
        
    Returns:
        The string name of the target node ("anchor_selector" or "generate_sql").
    """
    if state.get("is_complex"):
        return "anchor_selector"
    return "generate_sql"


def decomposer_router(state: State) -> List[Send]:
    """
    Orchestrates parallel map-reduce dispatching of Worker nodes via Send payloads.
    
    Args:
        state: The current global execution State.
        
    Returns:
        A list of LangGraph Send objects targeting the 'worker' node.
    """
    sub_tasks: List[Dict[str, Any]] = state.get("sub_tasks", [])
    # Scatter: dispatch an independent worker invocation for each sub-task in parallel
    return [Send("worker", {"current_task": task}) for task in sub_tasks]


def healing_router(state: State) -> Literal["heal_sql", "format_response"]:
    """
    Evaluates database errors to trigger self-healing loops or format final outputs.
    
    Args:
        state: The current global execution State.
        
    Returns:
        The string name of the next node ("heal_sql" or "format_response").
    """
    if state.get("sql_error"):
        if state.get("retry_count", 0) < 3:
            return "heal_sql"
    return "format_response"


# ### --- GRAPH BUILDER SECTION --- ###

def build_chatbot_graph(checkpointer: Optional[PostgresSaver] = None) -> CompiledStateGraph:
    """
    Compiles and constructs the LangGraph multi-agent StateGraph with persistence.
    
    Args:
        checkpointer: Optional external PostgresSaver. If None, instantiates a default.
        
    Returns:
        A compiled and ready CompiledStateGraph instance.
    """
    pool = get_connection_pool()
    if checkpointer is None:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
    
    # Establish persistent vector memory store using PostgreSQL
    store: PostgresStore = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["$"]
        }
    )
    store.setup()

    builder: StateGraph = StateGraph(State)

    # 1. Register Nodes
    builder.add_node("guardian", guardian_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("anchor_selector", anchor_selector_node)
    builder.add_node("column_pruner", column_pruner_node)
    builder.add_node("decomposer", decomposer_node)
    builder.add_node("worker", worker_node)
    builder.add_node("assembler", assembler_node)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("execute_sql", execute_sql_node)
    builder.add_node("heal_sql", heal_sql_node)
    builder.add_node("format_response", format_sql_response_node)

    # 2. Wire Control Flow Structure
    builder.add_edge(START, "guardian")
    
    # Guard routing
    builder.add_conditional_edges("guardian", guardian_router, {
        "classifier": "classifier",
        "clarify": "clarify",
        "end": END
    })

    builder.add_edge("clarify", END)

    # Complexity classifier routing
    builder.add_conditional_edges("classifier", classifier_router, {
        "anchor_selector": "anchor_selector",
        "generate_sql": "generate_sql"
    })

    # Divide & Conquer Map-Reduce Path (Complex Path)
    builder.add_edge("anchor_selector", "column_pruner")
    builder.add_edge("column_pruner", "decomposer")
    builder.add_conditional_edges("decomposer", decomposer_router, ["worker"])
    builder.add_edge("worker", "assembler")
    builder.add_edge("assembler", "execute_sql")

    # Simple Direct Path
    builder.add_edge("generate_sql", "execute_sql")

    # Execution and Healing Loop Router
    builder.add_conditional_edges("execute_sql", healing_router, {
        "heal_sql": "heal_sql",
        "format_response": "format_response"
    })

    builder.add_edge("heal_sql", "execute_sql")
    builder.add_edge("format_response", END)

    # Compile the complete graph with transaction savers and memory index stores
    return builder.compile(checkpointer=checkpointer, store=store)


# Compile a shared, singleton instance of the chatbot execution graph
sql_chatbot_graph: CompiledStateGraph = build_chatbot_graph()

