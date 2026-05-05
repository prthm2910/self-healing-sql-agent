from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from src.workflow.state import State
from src.workflow.nodes import (
    guardian_node,
    classifier_node,
    clarify_node,
    schema_selector_node,
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

def classifier_router(state: State) -> Literal["generate_sql", "schema_selector"]:
    """Routes based on query complexity."""
    if state.get("is_complex"):
        return "schema_selector"
    return "generate_sql"

def healing_router(state: State) -> Literal["heal_sql", "format_response"]:
    if state.get("sql_error"):
        if state.get("retry_count", 0) < 3:
            return "heal_sql"
    return "format_response"

def build_chatbot_graph():
    pool = get_connection_pool()
    checkpointer = PostgresSaver(pool)
    store = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["$"]
        }
    )
    
    checkpointer.setup()
    store.setup()

    builder = StateGraph(State)

    # 1. Add Nodes
    builder.add_node("guardian", guardian_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("schema_selector", schema_selector_node)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("execute_sql", execute_sql_node)
    builder.add_node("heal_sql", heal_sql_node)
    builder.add_node("format_response", format_sql_response_node)

    # --- Defined Logic ---
    builder.add_edge(START, "guardian")
    
    builder.add_conditional_edges("guardian", guardian_router, {
        "classifier": "classifier",
        "clarify": "clarify",
        "end": END
    })

    # Clarification node just asks the question and ends the turn
    builder.add_edge("clarify", END)

    builder.add_conditional_edges("classifier", classifier_router, {
        "schema_selector": "schema_selector",
        "generate_sql": "generate_sql"
    })

    builder.add_edge("schema_selector", "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")

    builder.add_conditional_edges("execute_sql", healing_router, {
        "heal_sql": "heal_sql",
        "format_response": "format_response"
    })

    builder.add_edge("heal_sql", "execute_sql")
    builder.add_edge("format_response", END)

    return builder.compile(checkpointer=checkpointer, store=store)

sql_chatbot_graph = build_chatbot_graph()
