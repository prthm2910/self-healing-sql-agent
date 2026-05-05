from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from src.workflow.state import State
from src.workflow.nodes import (
    guardian_node,
    call_chatbot,
    generate_sql_node, 
    execute_sql_node, 
    heal_sql_node, 
    format_sql_response_node
)
from src.services.database import get_connection_pool
from src.services.embeddings import get_embeddings_provider
from src.core.config import settings

def main_router(state: State) -> Literal["generate_sql", "chatbot", "end"]:
    """
    Primary router based on Guardian intent.
    """
    intent = state.get("intent", "chat")
    
    if intent == "blocked" or intent == "irrelevant":
        return "end"
    
    if intent == "sql":
        return "generate_sql"
        
    return "chatbot"

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
    
    # 2. Add Existing Nodes
    builder.add_node("chatbot", call_chatbot)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("execute_sql", execute_sql_node)
    builder.add_node("heal_sql", heal_sql_node)
    builder.add_node("format_response", format_sql_response_node)

    # --- Defined Logic ---
    builder.add_edge(START, "guardian")
    
    builder.add_conditional_edges("guardian", main_router, {
        "generate_sql": "generate_sql",
        "chatbot": "chatbot",
        "end": END
    })

    builder.add_edge("chatbot", END)
    builder.add_edge("generate_sql", "execute_sql")

    builder.add_conditional_edges("execute_sql", healing_router, {
        "heal_sql": "heal_sql",
        "format_response": "format_response"
    })

    builder.add_edge("heal_sql", "execute_sql")
    builder.add_edge("format_response", END)

    return builder.compile(checkpointer=checkpointer, store=store)

sql_chatbot_graph = build_chatbot_graph()
