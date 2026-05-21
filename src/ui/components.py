# ### --- IMPORTS --- ###
import uuid
from datetime import datetime
from typing import List, Dict, Any, Union
import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.core.config import settings
from src.services.database import delete_thread_data
from src.services.lessons import list_all_lessons

# ##############################################################################
# [Elaborative Breakdown] Streamlit Session State, Re-runs, and Component Hydration
# Why this sidebar design?
# Streamlit relies on a top-down script execution model where any user interaction (button 
# click, input entry) triggers a complete script re-run from the first line to the last. 
# Managing persistent state (such as dynamic conversation threads and vector search lessons)
# in this environment requires strict session state hydration protocols.
#
# Patterns & Hydration:
# 1. Thread Listing and Dynamic Buttons:
#    Threads are dynamically indexed via `PostgresStore` and sorted by update timestamp. 
#    Dynamic action buttons (deleting/selecting threads) leverage unique key strings 
#    (e.g., `del_{thread_id}`) to prevent button registration collissions during page 
#    re-runs.
# 2. Dynamic Deletion & Rerun Syncing:
#    When deleting a thread, database tables are cleared and the sidebar state is updated.
#    We then invoke `st.toast` and `st.rerun()` to immediately flush Streamlit's internal
#    render tree, forcing the parent layout to hydrate with the latest active values.
# 3. Lesson Expander Hydration:
#    Lessons distilled from historical self-healing debug sessions are rendered inside 
#    accordions (`st.expander`), giving the user immediate visual visibility into the
#    system's learning curve without cluttering the primary dialogue view.
# ##############################################################################


# ### --- SIDEBAR RENDER SECTION --- ###

def render_sidebar(chatbot_graph: CompiledStateGraph) -> None:
    """
    Renders the sidebar interface containing conversation threads and self-healing memory lessons.
    
    Args:
        chatbot_graph: Compiled LangGraph conversational state graph object.
        
    Returns:
        None (renders interface components directly in the Streamlit sidebar).
    """
    with st.sidebar:
        st.header("💬 Conversations")
        
        # Action button to clear active dialogue state and spin up a new chat session
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_thread_id = str(uuid.uuid4())
            st.rerun()
        
        st.divider()
        
        # --- Thread Management ---
        threads: List[Dict[str, Any]] = _get_all_threads(st.session_state.user_id, chatbot_graph)
        if threads:
            st.caption("Recent Chats")
            for thread in threads:
                col1, col2 = st.columns([0.8, 0.2])
                is_current: bool = thread["id"] == st.session_state.current_thread_id
                
                with col1:
                    # Select specific active thread conversation
                    if st.button(
                        thread["name"], 
                        key=f"select_{thread['id']}", 
                        use_container_width=True, 
                        type="primary" if is_current else "secondary"
                    ):
                        st.session_state.current_thread_id = thread["id"]
                        st.rerun()
                        
                with col2:
                    # Delete thread records from persistent store
                    if st.button("🗑️", key=f"del_{thread['id']}", help="Delete this thread"):
                        delete_thread_data(st.session_state.user_id, thread["id"], chatbot_graph.store)
                        if is_current:
                            st.session_state.current_thread_id = str(uuid.uuid4())
                        st.toast("Thread deleted.")
                        st.rerun()
        
        st.divider()

        # --- Lessons Management ---
        st.header("📖 Lessons")
        lessons: List[Dict[str, Any]] = list_all_lessons(chatbot_graph.store)
        if lessons:
            for lsn in lessons:
                icon: str = "📌" if lsn.get("type") == "pinned" else "🔄"
                with st.expander(f"{icon} {lsn.get('title', 'Untitled Lesson')}"):
                    st.markdown(f"**Instruction:** {lsn.get('instruction')}")
                    st.markdown(f"**Mistake:** {lsn.get('mistake')}")
                    st.caption(f"Reasoning: {lsn.get('reasoning')}")
        else:
            st.info("No lessons recorded yet.")
        
        st.divider()
        st.caption(f"User: {st.session_state.user_id}")
        if "request_timestamps" in st.session_state:
            st.caption(
                f"Requests (60s): {len(st.session_state.request_timestamps)}/{settings.rate_limit_rpm}"
            )


# ### --- DATA MANAGEMENT HELPER SECTION --- ###

def _get_all_threads(user_id: str, chatbot_graph: CompiledStateGraph) -> List[Dict[str, Any]]:
    """
    Retrieves all conversation threads associated with a given user ID from PostgreSQL store.
    
    Args:
        user_id: Target user identifier string.
        chatbot_graph: Compiled LangGraph object holding memory database configurations.
        
    Returns:
        A sorted list of thread dictionary metadata, ordered by last update time descending.
    """
    try:
        threads = chatbot_graph.store.search((user_id, "threads"), limit=100)
        return sorted(
            [
                {
                    "id": t.key, 
                    "name": t.value.get("name", t.key[:8]), 
                    "updated": t.value.get("updated")
                } 
                for t in threads
            ],
            key=lambda x: x["updated"] or "",
            reverse=True
        )
    except Exception as e:
        # Gracefully handle vector/store query failures without disrupting sidebar rendering
        return []


def save_thread_metadata(
    user_id: str, 
    thread_id: str, 
    chatbot_graph: CompiledStateGraph, 
    first_msg: str = "New Chat"
) -> None:
    """
    Saves or updates thread context metadata in the database transaction log.
    
    Args:
        user_id: Target user identifier string.
        thread_id: Target session thread identifier string.
        chatbot_graph: Compiled LangGraph chatbot object.
        first_msg: The first human message text used to summarize the thread description.
        
    Returns:
        None (persists data directly in the SQL vector store).
    """
    existing = chatbot_graph.store.get((user_id, "threads"), thread_id)
    name: str = (
        existing.value.get("name") 
        if existing and existing.value 
        else (first_msg[:30] + ("..." if len(first_msg) > 30 else ""))
    )
    chatbot_graph.store.put(
        (user_id, "threads"), 
        thread_id, 
        {"name": name, "updated": datetime.now().isoformat()}
    )

