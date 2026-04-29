from langgraph.graph import StateGraph, START, END
from src.workflow.state import State
from src.workflow.nodes import (
    call_chatbot,
    generate_sql_node, 
    execute_sql_node, 
    heal_sql_node, 
    format_sql_response_node
)
from typing import Literal

def main_router(state: State) -> Literal["generate_sql", "chatbot"]:
    """
    Primary router to decide between SQL analysis and standard chat.
    """
    last_message = state["messages"][-1].content.lower()
    # Keywords that trigger SQL analysis
    sql_keywords = ["actor", "film", "customer", "rental", "database", "how many", "list all", "table"]
    
    if any(k in last_message for k in sql_keywords):
        return "generate_sql"
    return "chatbot"

def healing_router(state: State) -> Literal["heal_sql", "format_response"]:
    """
    Routes based on SQL execution results.
    """
    if state.get("sql_error"):
        if state.get("retry_count", 0) < 3:
            return "heal_sql"
    return "format_response"

# Build Unified Graph
builder = StateGraph(State)

# Add Nodes
builder.add_node("chatbot", call_chatbot)
builder.add_node("generate_sql", generate_sql_node)
builder.add_node("execute_sql", execute_sql_node)
builder.add_node("heal_sql", heal_sql_node)
builder.add_node("format_response", format_sql_response_node)

# Define Logic
builder.add_conditional_edges(START, main_router, {
    "generate_sql": "generate_sql",
    "chatbot": "chatbot"
})

builder.add_edge("chatbot", END)

builder.add_edge("generate_sql", "execute_sql")

builder.add_conditional_edges("execute_sql", healing_router, {
    "heal_sql": "heal_sql",
    "format_response": "format_response"
})

builder.add_edge("heal_sql", "execute_sql")
builder.add_edge("format_response", END)

# Compile
sql_chatbot_graph = builder.compile()
