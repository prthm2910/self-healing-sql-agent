import uuid
import streamlit as st
from datetime import datetime

from src.core.config import settings
from src.services.database import delete_thread_data
from src.services.lessons import list_all_lessons

def render_sidebar(chatbot_graph):
    """Renders the conversation and memory management sidebar."""
    with st.sidebar:
        st.header("💬 Conversations")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_thread_id = str(uuid.uuid4())
            st.rerun()
        
        st.divider()
        
        # --- Thread Management ---
        threads = _get_all_threads(st.session_state.user_id, chatbot_graph)
        if threads:
            st.caption("Recent Chats")
            for thread in threads:
                col1, col2 = st.columns([0.8, 0.2])
                is_current = thread["id"] == st.session_state.current_thread_id
                with col1:
                    if st.button(thread["name"], key=f"select_{thread['id']}", use_container_width=True, type="primary" if is_current else "secondary"):
                        st.session_state.current_thread_id = thread["id"]
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{thread['id']}", help="Delete this thread"):
                        delete_thread_data(st.session_state.user_id, thread["id"], chatbot_graph.store)
                        if is_current:
                            st.session_state.current_thread_id = str(uuid.uuid4())
                        st.toast("Thread deleted.")
                        st.rerun()
        
        st.divider()

        # --- Lessons Management ---
        st.header("📖 Lessons")
        lessons = list_all_lessons(chatbot_graph.store)
        if lessons:
            for lsn in lessons:
                icon = "📌" if lsn["type"] == "pinned" else "🔄"
                with st.expander(f"{icon} {lsn['title']}"):
                    st.markdown(f"**Instruction:** {lsn['instruction']}")
                    st.markdown(f"**Mistake:** {lsn['mistake']}")
                    st.caption(f"Reasoning: {lsn['reasoning']}")
        else:
            st.info("No lessons recorded yet.")

        st.divider()
        st.caption(f"User: {st.session_state.user_id}")
        if "request_timestamps" in st.session_state:
            st.caption(f"Requests (60s): {len(st.session_state.request_timestamps)}/{settings.rate_limit_rpm}")

def _get_all_threads(user_id, chatbot_graph):
    try:
        threads = chatbot_graph.store.search((user_id, "threads"), limit=100)
        return sorted(
            [{"id": t.key, "name": t.value.get("name", t.key[:8]), "updated": t.value.get("updated")} for t in threads],
            key=lambda x: x["updated"] or "",
            reverse=True
        )
    except Exception:
        return []

def save_thread_metadata(user_id, thread_id, chatbot_graph, first_msg="New Chat"):
    existing = chatbot_graph.store.get((user_id, "threads"), thread_id)
    name = existing.value.get("name") if existing else (first_msg[:30] + ("..." if len(first_msg) > 30 else ""))
    chatbot_graph.store.put((user_id, "threads"), thread_id, {"name": name, "updated": datetime.now().isoformat()})
