import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import settings
from src.ui.components import render_sidebar, save_thread_metadata
from src.utils.limiter import rate_limiter
from src.workflow.builder import build_chatbot_graph
from src.utils.logger import logger, log_context
from langsmith import Client

# Initialize LangSmith client for tracing
ls_client = Client()

# Page Config
st.set_page_config(page_title="Personalized AI Assistant", page_icon="🧠", layout="wide")

# Main Header
st.title("🧠 Personalized AI Assistant")
st.markdown("Persistent long-term memory via Neon DB & LangGraph Store (SOLID Refactor).")

# --- Initialize Session State ---
if "user_id" not in st.session_state:
    st.session_state.user_id = settings.default_user_id
    logger.info(f"Initialized session for User: {st.session_state.user_id}")

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())
    logger.info(f"Generated new thread ID: {st.session_state.current_thread_id}")

if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Update log context for UI thread
log_context.set({
    "user_id": st.session_state.user_id,
    "thread_id": st.session_state.current_thread_id
})

# --- Build Graph ---
@st.cache_resource
def get_graph():
    """Returns the compiled LangGraph workflow."""
    logger.info("Building/loading chatbot graph...")
    return build_chatbot_graph()

# Module-level Constant for the Graph
CHATBOT_GRAPH = get_graph()

# --- Sidebar ---
render_sidebar(CHATBOT_GRAPH)

# --- Chat Configuration ---
config = {
    "configurable": {
        "thread_id": st.session_state.current_thread_id,
        "user_id": st.session_state.user_id
    }
}

# --- Display Messages ---
chat_container = st.container()

with chat_container:
    try:
        state = CHATBOT_GRAPH.get_state(config)
        messages = state.values.get("messages", [])
    except Exception as e:
        logger.error(f"Error loading chat history: {e}", exc_info=True)
        st.error(f"Error loading chat history: {e}")
        messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            clean_content = msg.content.replace(settings.memory_tag, "").strip()
            if clean_content:
                with st.chat_message("assistant"):
                    st.markdown(clean_content)

# --- Chat Input ---
prompt = st.chat_input(
    "What's on your mind?", 
    disabled=st.session_state.is_thinking
)

# Phase 1: Input Detection & Immediate Lock
if prompt:
    logger.info(f"User interaction: prompt received (len={len(prompt)})")
    # 1. Rate Limit Check (Shared Global Limit)
    if rate_limiter.get_current_load() >= settings.rate_limit_rpm:
        logger.warning(f"Rate limit hit: {settings.rate_limit_rpm} RPM")
        st.error(f"⚠️ Rate limit reached ({settings.rate_limit_rpm} RPM). Please wait.")
        st.stop()
    
    # 2. Lock UI and capture prompt
    st.session_state.is_thinking = True
    st.session_state.pending_prompt = prompt
    st.rerun()

# Phase 2: Processing the Pending Prompt
if st.session_state.pending_prompt:
    current_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None # Clear immediately to prevent loops

    # 3. Handle First Message (Thread Indexing)
    if not messages:
        logger.info(f"First message in thread {st.session_state.current_thread_id}. Saving metadata.")
        save_thread_metadata(
            st.session_state.user_id, 
            st.session_state.current_thread_id, 
            CHATBOT_GRAPH, 
            current_prompt
        )
    
    # 4. Show User Message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(current_prompt)

    # 5. Invoke Graph (Within a spinner)
    with st.chat_message("assistant"):
        with st.spinner("Searching memory & thinking..."):
            try:
                logger.info(f"Invoking chatbot graph for thread {st.session_state.current_thread_id}")
                input_data = {
                    "messages": [HumanMessage(content=current_prompt)], 
                    "user_id": st.session_state.user_id
                }
                response = CHATBOT_GRAPH.invoke(input_data, config=config)
                
                # Update thread metadata (timestamp)
                save_thread_metadata(
                    st.session_state.user_id, 
                    st.session_state.current_thread_id, 
                    CHATBOT_GRAPH, 
                    current_prompt
                )

                # Get and display response
                ai_msg = response["messages"][-1]
                content = ai_msg.content
                display_content = content.replace(settings.memory_tag, "").strip()
                st.markdown(display_content)
                logger.info("Response received and displayed.")
            
            except Exception as e:
                logger.error(f"AI Invocation Error: {e}", exc_info=True)
                st.error(f"AI Error: {e}")
            
            finally:
                # 7. Unlock Input and Rerun to refresh the field
                st.session_state.is_thinking = False
                st.rerun()
