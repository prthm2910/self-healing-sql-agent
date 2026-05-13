from typing import Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from src.workflow.state import State
from src.workflow.nodes import (
    guardian_node,
    classifier_node,
    clarify_node,
    anchor_selector_node,
    column_pruner_node,
    decomposer_node,
    worker_node,
    assembler_node,
    generate_sql_node, 
    execute_sql_node, 
    heal_sql_node, 
    format_sql_response_node
)
from src.services.database import get_connection_pool
from src.services.embeddings import get_embeddings_provider
from src.core.config import settings

def guardian_router(state: State) -> Literal["classifier", "clarify", "end"]:
    """
    Primary router based on Guardian intent.
    """
    intent = state.get("intent", "DENY")
    if intent == "SQL":
        return "classifier"
    if intent == "CLARIFY":
        return "clarify"
    return "end"

def classifier_router(state: State) -> Literal["generate_sql", "anchor_selector"]:
    """Routes based on query complexity."""
    if state.get("is_complex"):
        return "anchor_selector"
    return "generate_sql"

def decomposer_router(state: State):
    """Parallel Send router for Worker Nodes."""
    sub_tasks = state.get("sub_tasks", [])
    # Dispatch workers in parallel
    return [Send("worker", {"current_task": task}) for task in sub_tasks]

def healing_router(state: State) -> Literal["heal_sql", "format_response"]:
    if state.get("sql_error"):
        if state.get("retry_count", 0) < 3:
            return "heal_sql"
    return "format_response"

def build_chatbot_graph(checkpointer=None):
    pool = get_connection_pool()
    if checkpointer is None:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
    
    store = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["$"]
        }
    )
    store.setup()

    builder = StateGraph(State)

    # 1. Add Nodes
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

    # 2. Define Logic
    builder.add_edge(START, "guardian")
    
    builder.add_conditional_edges("guardian", guardian_router, {
        "classifier": "classifier",
        "clarify": "clarify",
        "end": END
    })

    builder.add_edge("clarify", END)

    builder.add_conditional_edges("classifier", classifier_router, {
        "anchor_selector": "anchor_selector",
        "generate_sql": "generate_sql"
    })

    # Complex Path (Divide & Conquer)
    builder.add_edge("anchor_selector", "column_pruner")
    builder.add_edge("column_pruner", "decomposer")
    builder.add_conditional_edges("decomposer", decomposer_router, ["worker"])
    builder.add_edge("worker", "assembler")
    builder.add_edge("assembler", "execute_sql")

    # Simple Path
    builder.add_edge("generate_sql", "execute_sql")

    # Execution & Healing
    builder.add_conditional_edges("execute_sql", healing_router, {
        "heal_sql": "heal_sql",
        "format_response": "format_response"
    })

    builder.add_edge("heal_sql", "execute_sql")
    builder.add_edge("format_response", END)

    return builder.compile(checkpointer=checkpointer, store=store)

sql_chatbot_graph = build_chatbot_graph()
