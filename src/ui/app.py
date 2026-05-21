# ### --- IMPORTS --- ###
import uuid
from typing import List, Dict, Any, Union, Optional
import streamlit as st
from langsmith import Client
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph.state import CompiledStateGraph

from src.core.config import settings
from src.utils.limiter import rate_limiter
from src.utils.logger import logger, log_context
from src.workflow.builder import build_chatbot_graph
from src.ui.components import render_sidebar, save_thread_metadata

# ##############################################################################
# [Elaborative Breakdown] Streamlit Reactive Execution, Input Intercepts, & Context Logging
# Why this application lifecycle layout?
# Streamlit executes top-down on every state change. If an LLM invocation takes 5 seconds,
# a user could double-click or submit multiple chats concurrently. This could lead to
# duplicate API calls, thread corruption, or database pool exhaustions.
#
# Execution & Control Flow Lifecycle Phases:
#
# 1. State Initialization & Log Synchronization:
#    During boot, we hydrate the `user_id` and `current_thread_id` in `st.session_state`.
#    We immediately sync these parameters to the python `log_context` ContextVar. This guarantees
#    that all downstream operations, database queries, and vector store operations inherit the 
#    matching session attributes in active thread logging.
# 2. Page Caching (`@st.cache_resource`):
#    Compiling the LangGraph StateGraph (incorporating DB connections, Vector stores, and models)
#    is expensive. We cache this globally via `get_graph()`, keeping compilation overhead to a
#    single execution during app boot.
# 3. Phase 1: Input Interception and State Locking:
#    When `st.chat_input` registers an input, we immediately intercept the call, check the 
#    sliding window rate limiter, and lock the UI:
#    `st.session_state.is_thinking = True`
#    `st.session_state.pending_prompt = prompt`
#    We then invoke `st.rerun()`. This terminates execution instantly and restarts top-down. 
#    On the re-run, the text input widget is cleanly rendered as disabled to prevent concurrent 
#    submissions.
# 4. Phase 2: Orchestration & Session Resumption:
#    During the locked re-run, we pull the cached `pending_prompt`, render it instantly, invoke
#    the CompiledStateGraph within an active spinner context, persist metadata to database tables,
#    and unlock the UI state.
# ##############################################################################


# ### --- PAGE CONFIGURATION SECTION --- ###

st.set_page_config(
    page_title="Personalized AI Assistant", 
    page_icon="🧠", 
    layout="wide"
)

st.title("🧠 Personalized AI Assistant")
st.markdown("Persistent long-term memory via Neon DB & LangGraph Store (SOLID Refactor).")


# ### --- SESSION STATE HYDRATION --- ###

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

# Synchronize context attributes for the dynamic logging filters
log_context.set({
    "user_id": str(st.session_state.user_id),
    "thread_id": str(st.session_state.current_thread_id)
})


# ### --- GRAPH RESOURCE CACHING --- ###

@st.cache_resource
def get_graph() -> CompiledStateGraph:
    """
    Retrieves or compiles the shared, singleton LangGraph StateGraph instance.
    
    Returns:
        The compiled, persists-backed CompiledStateGraph workflow.
    """
    logger.info("Building/loading chatbot graph...")
    return build_chatbot_graph()


# Module-level compiled graph constant
CHATBOT_GRAPH: CompiledStateGraph = get_graph()


# ### --- SIDEBAR AND LAYOUT --- ###

render_sidebar(CHATBOT_GRAPH)

# Session configurations for thread tracking
config: Dict[str, Any] = {
    "configurable": {
        "thread_id": st.session_state.current_thread_id,
        "user_id": st.session_state.user_id
    }
}


# ### --- CHAT MESSAGE HISTORY RENDER --- ###

chat_container = st.container()

with chat_container:
    try:
        # Retrieve the latest persistent state snapshot from database savers
        graph_state = CHATBOT_GRAPH.get_state(config)
        messages: List[BaseMessage] = graph_state.values.get("messages", [])
    except Exception as e:
        logger.error(f"Error loading chat history: {e}", exc_info=True)
        st.error(f"Error loading chat history: {e}")
        messages = []

    # Map message logs into corresponding human and bot message styles
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            # Clean special trigger flags from presenting on frontend view
            clean_content: str = msg.content.replace(settings.memory_tag, "").strip()
            if clean_content:
                with st.chat_message("assistant"):
                    st.markdown(clean_content)


# ### --- CHAT INPUT REGISTRATION --- ###

prompt: Optional[str] = st.chat_input(
    "What's on your mind?", 
    disabled=st.session_state.is_thinking
)


# ### --- PHASE 1: LOCK AND VALIDATE --- ###

if prompt:
    logger.info(f"User interaction: prompt received (len={len(prompt)})")
    
    # 1. Evaluate current sliding window requests for rate limit constraints
    current_rpm: int = rate_limiter.get_stats().get("rpm", 0)
    if current_rpm >= settings.rate_limit_rpm:
        logger.warning(f"Rate limit hit: {settings.rate_limit_rpm} RPM")
        st.error(f"⚠️ Rate limit reached ({settings.rate_limit_rpm} RPM). Please wait.")
        st.stop()
    
    # 2. Lock widgets and transition prompt to state variables
    st.session_state.is_thinking = True
    st.session_state.pending_prompt = prompt
    st.rerun()


# ### --- PHASE 2: PROCESSING CYCLES --- ###

if st.session_state.pending_prompt:
    current_prompt: str = st.session_state.pending_prompt
    st.session_state.pending_prompt = None  # Flush immediately to prevent recursive loop runs

    # 1. Index metadata if this marks the first message in the session thread
    if not messages:
        logger.info(f"First message in thread {st.session_state.current_thread_id}. Saving metadata.")
        save_thread_metadata(
            st.session_state.user_id, 
            st.session_state.current_thread_id, 
            CHATBOT_GRAPH, 
            current_prompt
        )
    
    # 2. Render prompt instantly in user context
    with chat_container:
        with st.chat_message("user"):
            st.markdown(current_prompt)

    # 3. Compile context and invoke the LangGraph execution flow
    with st.chat_message("assistant"):
        with st.spinner("Searching memory & thinking..."):
            try:
                logger.info(f"Invoking chatbot graph for thread {st.session_state.current_thread_id}")
                input_data: Dict[str, Any] = {
                    "messages": [HumanMessage(content=current_prompt)], 
                    "user_id": st.session_state.user_id
                }
                
                # Execute graph traversal
                response = CHATBOT_GRAPH.invoke(input_data, config=config)
                
                # Update thread interaction timestamp
                save_thread_metadata(
                    st.session_state.user_id, 
                    st.session_state.current_thread_id, 
                    CHATBOT_GRAPH, 
                    current_prompt
                )

                # Format response content
                ai_msg = response["messages"][-1]
                content: str = ai_msg.content
                display_content: str = content.replace(settings.memory_tag, "").strip()
                st.markdown(display_content)
                logger.info("Response received and displayed.")
            
            except Exception as e:
                logger.error(f"AI Invocation Error: {e}", exc_info=True)
                st.error(f"AI Error: {e}")
            
            finally:
                # 4. Release locks and re-run to reset fields
                st.session_state.is_thinking = False
                st.rerun()
st.rerun()
