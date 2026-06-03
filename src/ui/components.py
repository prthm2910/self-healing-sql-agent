# ### --- IMPORTS --- ###
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from langgraph.graph.state import CompiledStateGraph

from src.core.config import settings
from src.services.database import delete_thread_data
from src.services.lessons import list_all_lessons


# ##############################################################################
# [Elaborative Breakdown] Gradio-Compatible Data Helpers
# Why pure data functions?
# Gradio uses event-driven callbacks rather than Streamlit's top-down re-execution.
# These helpers return raw data (lists, dicts) that Gradio components consume.
# No st.sidebar, st.button, or st.rerun — the Gradio layer handles UI state.
# ##############################################################################


# ### --- THREAD MANAGEMENT --- ###


def _get_all_threads(
    user_id: str,
    chatbot_graph: CompiledStateGraph,
) -> List[Dict[str, Any]]:
    """
    Retrieves all conversation threads for a user from the PostgreSQL store.

    Args:
        user_id: Target user identifier.
        chatbot_graph: Compiled LangGraph object holding the PostgresStore.

    Returns:
        Sorted list of thread metadata dicts, newest first.
    """
    try:
        threads = chatbot_graph.store.search((user_id, "threads"), limit=100)
        return sorted(
            [
                {
                    "id": t.key,
                    "name": t.value.get("name", t.key[:8]),
                    "updated": t.value.get("updated") or "",
                }
                for t in threads
            ],
            key=lambda x: x["updated"],
            reverse=True,
        )
    except Exception:
        return []


def save_thread_metadata(
    user_id: str,
    thread_id: str,
    chatbot_graph: CompiledStateGraph,
    first_msg: str = "New Chat",
) -> None:
    """
    Saves or updates thread metadata in the PostgreSQL store.

    Args:
        user_id: Target user identifier.
        thread_id: Session thread identifier.
        chatbot_graph: Compiled LangGraph object.
        first_msg: First human message used to summarise the thread name.
    """
    existing = chatbot_graph.store.get((user_id, "threads"), thread_id)
    name = (
        existing.value.get("name")
        if existing and existing.value
        else (first_msg[:30] + ("..." if len(first_msg) > 30 else ""))
    )
    chatbot_graph.store.put(
        (user_id, "threads"),
        thread_id,
        {"name": name, "updated": datetime.now().isoformat()},
    )


def get_threads_table(
    user_id: str,
    chatbot_graph: CompiledStateGraph,
) -> List[List[str]]:
    """
    Returns threads formatted as a 2D list for gr.Dataframe.

    Each row: [thread_name, thread_id, last_updated]
    """
    threads = _get_all_threads(user_id, chatbot_graph)
    if not threads:
        return []
    return [
        [t["name"], t["id"], _format_timestamp(t["updated"])]
        for t in threads
    ]


def _format_timestamp(iso_str: str) -> str:
    """Convert ISO datetime to a short, readable format."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %H:%M")
    except (ValueError, TypeError):
        return iso_str


# ### --- LESSONS --- ###


def get_lessons_data(
    chatbot_graph: CompiledStateGraph,
) -> List[Dict[str, str]]:
    """
    Returns all lessons formatted for left-panel accordion rendering.

    Returns:
        List of dicts with keys: icon, title, instruction, mistake, reasoning.
    """
    lessons = list_all_lessons(chatbot_graph.store)
    result = []
    for lsn in lessons:
        result.append({
            "icon": "📌" if lsn.get("type") == "pinned" else "🔄",
            "title": lsn.get("title", "Untitled Lesson"),
            "instruction": lsn.get("instruction", ""),
            "mistake": lsn.get("mistake", ""),
            "reasoning": lsn.get("reasoning", ""),
        })
    return result


# ### --- CHAT HISTORY --- ###


def load_chat_history(
    chatbot_graph: CompiledStateGraph,
    thread_id: str,
    user_id: str,
) -> List[Dict[str, str]]:
    """
    Loads the conversation history for a given thread from the LangGraph state.

    Returns:
        List of message dicts compatible with gr.Chatbot(type="messages"):
        [{"role": "user"/"assistant", "content": "..."}]
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }
    try:
        graph_state = chatbot_graph.get_state(config)
        messages = graph_state.values.get("messages", [])
        history = []
        for msg in messages:
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            content = msg.content
            if role == "assistant":
                content = content.replace(settings.memory_tag, "").strip()
            if content:
                history.append({"role": role, "content": content})
        return history
    except Exception:
        return []
