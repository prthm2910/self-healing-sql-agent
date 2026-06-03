# ### --- IMPORTS --- ###
import uuid
import time
import json
import warnings
from typing import List, Dict, Any, Optional

import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph

# Suppress noisy but benign structured-output warnings from NVIDIA NIM SDK
warnings.filterwarnings(
    "ignore",
    message=".*is not known to support structured output.*",
    category=UserWarning
)

from src.core.config import settings
from src.utils.limiter import rate_limiter
from src.utils.logger import logger, log_context
from src.workflow.builder import build_chatbot_graph
from src.services.llm import get_llm
from src.ui.components import (
    get_threads_table,
    get_lessons_data,
    load_chat_history,
    save_thread_metadata,
)


# ##############################################################################
# [Elaborative Breakdown] Gradio Event-Driven Chat Architecture
# Why this design?
# Gradio uses async event callbacks rather than Streamlit's top-down re-execution.
# We chain events with .then():
#   1. add_message() [queue=False] → instantly adds user message, clears input
#   2. bot() [generator] → streams thinking steps + final response progressively
#
# Streaming via graph.stream():
#   LangGraph's stream() yields (node_name, state_update) tuples as each node
#   completes. We map these to Gradio ChatMessage entries with nested metadata
#   (parent_id / id) to create collapsible accordions inside the chat bubble.
# ##############################################################################


# ### --- GRAPH RESOURCE CACHING --- ###

def _get_graph() -> CompiledStateGraph:
    """Returns the shared LangGraph StateGraph instance."""
    logger.info("Building/loading chatbot graph...")
    return build_chatbot_graph()


# Module-level compiled graph constant
CHATBOT_GRAPH: CompiledStateGraph = _get_graph()


# ### --- NODE TITLE MAPPING --- ###

NODE_STEP_MAP: Dict[str, Dict[str, str]] = {
    "guardian":       {"title": "🔒 Validating intent...", "detail_key": "intent"},
    "classifier":     {"title": "📊 Classifying complexity...", "detail_key": "is_complex"},
    "clarify":        {"title": "💬 Asking for clarification...", "detail_key": None},
    "anchor_selector":{"title": "📐 Selecting relevant tables...", "detail_key": "selected_tables"},
    "column_pruner":  {"title": "✂️ Pruning schema columns...", "detail_key": "selected_columns"},
    "decomposer":     {"title": "🔧 Decomposing into sub-tasks...", "detail_key": "sub_tasks"},
    "worker":         {"title": "⚙️ Generating SQL snippets...", "detail_key": "current_task"},
    "assembler":      {"title": "🧩 Assembling final query...", "detail_key": "current_sql"},
    "execute_sql":    {"title": "🗃️ Executing SQL query...", "detail_key": "sql_results"},
    "heal_sql":       {"title": "🩹 Self-healing SQL error...", "detail_key": "sql_error"},
    "format_response":{"title": "📝 Formatting response...", "detail_key": None},
}


def _extract_detail(node_name: str, state_update: Dict[str, Any], elapsed: Optional[float] = None) -> str:
    """Extract a detailed, human-readable log string from a node's state update.

    Args:
        node_name: Name of the node that produced this output.
        state_update: The state dict returned by the node.
        elapsed: Seconds elapsed since the request started (for timing display).
    """
    mapping = NODE_STEP_MAP.get(node_name, {})
    detail_key = mapping.get("detail_key")

    parts = []

    if node_name == "guardian":
        intent = state_update.get("intent", "")
        parts.append(f"**Intent:** `{intent}`")
        logs = state_update.get("agent_logs", [])
        if logs and isinstance(logs, list):
            last_log = logs[-1] if logs else {}
            thought = last_log.get("thought_process", "")
            if thought:
                parts.append(f"**Thought:** {thought}")
        if intent == "DENY":
            msgs = state_update.get("messages", [])
            for m in msgs:
                if isinstance(m, AIMessage) and m.content:
                    parts.append(f"**Response:** {m.content}")
        return "\n\n".join(parts) if parts else ""

    elif node_name == "classifier":
        is_complex = state_update.get("is_complex", False)
        parts.append(f"**Complexity:** {'Complex (multi-table)' if is_complex else 'Simple (direct SQL)'}")
        return "\n\n".join(parts)

    elif node_name == "clarify":
        msgs = state_update.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage) and m.content:
                parts.append(f"**Question:** {m.content}")
        return "\n\n".join(parts) if parts else ""

    elif node_name == "anchor_selector":
        tables = state_update.get("selected_tables", [])
        parts.append(f"**Tables identified:** {', '.join(tables) if tables else 'None'}")
        return "\n\n".join(parts)

    elif node_name == "column_pruner":
        cols = state_update.get("selected_columns", {})
        if isinstance(cols, dict):
            total = sum(len(c) for c in cols.values())
            parts.append(f"**Columns:** {total} across {len(cols)} tables")
            for tbl, cl in cols.items():
                parts.append(f"- `{tbl}`: {', '.join(cl)}")
        return "\n\n".join(parts) if parts else ""

    elif node_name == "decomposer":
        tasks = state_update.get("sub_tasks", [])
        parts.append(f"**Sub-tasks:** {len(tasks)} identified")
        if isinstance(tasks, list):
            for i, t in enumerate(tasks, 1):
                desc = t.get("description", str(t)[:80]) if isinstance(t, dict) else str(t)[:80]
                parts.append(f"{i}. {desc}")
        return "\n\n".join(parts)

    elif node_name == "worker":
        task = state_update.get("current_task", {})
        task_desc = task.get("description", "processing") if isinstance(task, dict) else "processing"
        # Extract which sub-task number this is from decomposer
        sub_tasks = state_update.get("sub_tasks", [])
        task_index = ""
        if sub_tasks and isinstance(sub_tasks, list):
            idx = len(state_update.get("sql_snippets", {}))
            total = len(sub_tasks)
            task_index = f" (Task {idx + 1}/{total})"
        snippets = state_update.get("sql_snippets", {})
        parts.append(f"**Task:** {task_desc}{task_index}")
        if snippets:
            last_key = list(snippets.keys())[-1]
            sql = snippets[last_key]
            parts.append(f"**SQL:**\n```sql\n{sql[:200]}{'...' if len(sql) > 200 else ''}\n```")
        return "\n\n".join(parts)

    elif node_name == "assembler":
        sql = state_update.get("current_sql", "")
        if sql:
            parts.append(f"**Full Assembled SQL:**\n```sql\n{sql}\n```")
        return "\n\n".join(parts)

    elif node_name == "execute_sql":
        sql = state_update.get("current_sql", "")
        results = state_update.get("sql_results", [])
        error = state_update.get("sql_error", "")
        is_agg = state_update.get("is_aggregated", False)
        if sql:
            parts.append(f"**Query executed:**\n```sql\n{sql[:200]}{'...' if len(sql) > 200 else ''}\n```")
        if isinstance(results, list):
            parts.append(f"**Rows returned:** {len(results)}")
            if results and len(results) <= 5:
                parts.append(f"**Data:**\n```json\n{json.dumps(results, indent=2, default=str)[:300]}\n```")
        if error:
            parts.append(f"**Error:** {error[:200]}")
        if is_agg:
            parts.append("**Aggregated result**")
        return "\n\n".join(parts)

    elif node_name == "heal_sql":
        error = state_update.get("sql_error", "")
        retry = state_update.get("retry_count", 0)
        sql = state_update.get("current_sql", "")
        original_sql = state_update.get("original_sql", sql)
        parts.append(f"**Retry {retry}/3**")
        if original_sql:
            parts.append(f"**Original Query:**\n```sql\n{original_sql}\n```")
        if error:
            parts.append(f"**SQL Error:**\n`{error}`")
        if sql and sql != original_sql:
            parts.append(f"**Updated Query:**\n```sql\n{sql}\n```")
        return "\n\n".join(parts)

    elif node_name == "format_response":
        msgs = state_update.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage) and m.content:
                clean = m.content.replace(settings.memory_tag, "").strip()
                parts.append(f"**Response length:** {len(clean)} chars")
        return "\n\n".join(parts) if parts else ""

    # Fallback
    if detail_key and detail_key in state_update:
        return str(state_update[detail_key])[:200]
    return ""


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable duration string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.0f}s"


# ### --- FOLLOW-UP CONTEXT (LLM-BASED, NO HARD-CODED PATTERNS) --- ###


def _get_prior_context(thread_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve prior SQL results + recent messages from the thread state."""
    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id}
    }
    try:
        state = CHATBOT_GRAPH.get_state(config)
        vals = state.values if state else {}
        msgs = vals.get("messages", [])

        recent_user_msgs = []
        recent_assistant_msgs = []
        for m in reversed(msgs):
            if len(recent_assistant_msgs) >= 3 and len(recent_user_msgs) >= 3:
                break
            if isinstance(m, AIMessage):
                if len(recent_assistant_msgs) < 3:
                    recent_assistant_msgs.append(
                        m.content.replace(settings.memory_tag, "").strip()
                    )
            elif isinstance(m, HumanMessage):
                if len(recent_user_msgs) < 3:
                    recent_user_msgs.append(m.content)

        sql = vals.get("current_sql", "")
        results = vals.get("sql_results")
        if results and sql:
            return {
                "prior_sql": sql,
                "prior_results": results if isinstance(results, list) else [],
                "is_aggregated": vals.get("is_aggregated", False),
                "recent_user_msgs": list(reversed(recent_user_msgs)),
                "recent_assistant_msgs": list(reversed(recent_assistant_msgs)),
            }
    except Exception as e:
        logger.debug(f"No prior context: {e}")
    return None


def _classify_follow_up(message: str, prior_ctx: Optional[Dict[str, Any]]) -> str:
    """
    LLM-based classification: is this message asking ABOUT prior results
    (interpretive), or requesting NEW data (sql), or first turn (fresh)?
    """
    if prior_ctx is None:
        return "fresh"

    try:
        llm = get_llm()

        results = prior_ctx.get("prior_results", [])
        results_str = json.dumps(results[:5], default=str)[:400] if results else "N/A"
        recent_q = prior_ctx["recent_user_msgs"][-1] if prior_ctx["recent_user_msgs"] else ""

        resp = llm.invoke(
            f"You classify a user message into exactly one category.\n\n"
            f"PRIOR CONTEXT:\n"
            f"Last question: {recent_q}\n"
            f"SQL: {prior_ctx['prior_sql'][:200]}\n"
            f"Results: {results_str}\n\n"
            f"NEW MESSAGE: {message}\n\n"
            f"CATEGORIES:\n"
            f"INTERPRETIVE — Asking about/explaining the PRIOR results "
            f"(currency, units, meaning, why is this high, what does this represent, "
            f"what is the data type, which database, which schema, etc).\n"
            f"SQL — Wants a NEW query, different filter, different computation, or new data retrieval.\n"
            f"CHITCHAT — Greeting, thanks, small talk.\n\n"
            f"Reply with ONLY one word: INTERPRETIVE, SQL, or CHITCHAT."
        )
        answer = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()

        if "INTERPRETIVE" in answer:
            return "interpretive"
        if "CHITCHAT" in answer:
            return "chitchat"
        return "sql"

    except Exception as e:
        logger.error(f"Follow-up classification error: {e}")
        return "sql"  # safe fallback


def _answer_follow_up(
    message: str,
    prior_ctx: Dict[str, Any],
    history: List[Dict[str, Any]],
    parent_id: str,
) -> str:
    """Answer an interpretive follow-up using LLM with full prior context."""
    try:
        llm = get_llm()

        results = prior_ctx.get("prior_results", [])
        sql = prior_ctx.get("prior_sql", "")
        recent_q = prior_ctx["recent_user_msgs"][-1] if prior_ctx["recent_user_msgs"] else ""
        recent_a = prior_ctx["recent_assistant_msgs"][-1] if prior_ctx["recent_assistant_msgs"] else ""

        if prior_ctx.get("is_aggregated", False):
            results_detail = json.dumps(results, indent=2, default=str)
        else:
            cols = list(results[0].keys()) if results and isinstance(results[0], dict) else []
            rows_detail = json.dumps(results[:10], default=str)[:500]
            results_detail = (
                f"Columns: {', '.join(cols)}\n"
                f"Rows: {len(results)} total\n"
                f"Sample:\n{rows_detail}"
            )

        resp = llm.invoke(
            f"You are an AI assistant. The user received SQL results and is asking a follow-up.\n\n"
            f"IMPORTANT: Answer the user's question DIRECTLY. Do NOT rephrase their question.\n"
            f"Do NOT generate new SQL. Do NOT ask a new question to the database.\n"
            f"If you can answer from the context below, give a concise, direct answer.\n\n"
            f"PRIOR EXCHANGE:\n"
            f"User: {recent_q}\n"
            f"Assistant: {recent_a[:500]}\n\n"
            f"SQL: {sql}\n"
            f"Results:\n{results_detail}\n\n"
            f"USER FOLLOW-UP: {message}\n\n"
            f"Answer directly from the context above. Be concise. "
            f"If the context doesn't contain enough info, say what's missing."
        )
        answer = resp.content if hasattr(resp, "content") else str(resp)

        # Show a context-usage accordion step
        history.append({
            "role": "assistant",
            "content": (
                f"**Context used:**\n"
                f"- Prior query: {recent_q[:80]}\n"
                f"- Results: {len(results)} rows"
            ),
            "metadata": {
                "title": "🔍 Answering from prior context...",
                "id": f"{parent_id}_ctx",
                "parent_id": parent_id,
                "status": "done",
            },
        })
        return answer.strip()

    except Exception as e:
        logger.error(f"Follow-up answer error: {e}", exc_info=True)
        return f"I couldn't answer that from prior context. Error: {str(e)}"


# ### --- BOT GENERATOR (STREAMING + THINKING STEPS) --- ###


def bot(
    message: str,
    history: List[Dict[str, Any]],
    thread_id: str,
    user_id: str,
) -> Any:
    """
    Generator that streams the agent's response progressively.

    - Fresh / SQL follow-up → full graph.stream()
    - Interpretive follow-up → LLM answers from prior context (no SQL re-execution)
    """
    log_context.set({"user_id": str(user_id), "thread_id": str(thread_id)})

    # --- Rate limit check ---
    current_rpm = rate_limiter.get_stats().get("rpm", 0)
    if current_rpm >= settings.rate_limit_rpm:
        logger.warning(f"Rate limit hit: {settings.rate_limit_rpm} RPM")
        history.append(
            {"role": "assistant", "content": f"⚠️ Rate limit reached ({settings.rate_limit_rpm} RPM). Please wait."}
        )
        yield history
        return

    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id}
    }

    # --- LLM-based follow-up classification ---
    prior_ctx = _get_prior_context(thread_id, user_id)
    follow_up_type = _classify_follow_up(message, prior_ctx)
    logger.info(f"Follow-up type: {follow_up_type} | msg: {message}")

    # --- Parent thinking accordion (collapsed by default) ---
    parent_id = str(uuid.uuid4())
    history.append({
        "role": "assistant", "content": "",
        "metadata": {"title": "💭 Thought Process", "id": parent_id, "status": "done"},
    })
    yield history

    # --- Interpretive follow-up: answer from context, skip graph ---
    if follow_up_type == "interpretive" and prior_ctx:
        answer = _answer_follow_up(message, prior_ctx, history, parent_id)
        yield history

        history.append({"role": "assistant", "content": ""})
        for char in answer:
            history[-1]["content"] += char
            time.sleep(0.005)
            yield history

        try:
            save_thread_metadata(user_id, thread_id, CHATBOT_GRAPH, message)
        except Exception:
            pass
        return

    # --- Chitchat: quick LLM answer, skip graph ---
    if follow_up_type == "chitchat":
        try:
            llm = get_llm()
            resp = llm.invoke(f"Reply briefly and warmly: {message}")
            answer = resp.content if hasattr(resp, "content") else str(resp)
            history.append({"role": "assistant", "content": answer.strip()})
            yield history
        except Exception:
            pass
        try:
            save_thread_metadata(user_id, thread_id, CHATBOT_GRAPH, message)
        except Exception:
            pass
        return

    # --- Fresh or SQL follow-up: run the full graph ---
    input_data = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
    }

    try:
        final_response_text = ""
        step_counter = 0
        graph_start = time.time()

        for step in CHATBOT_GRAPH.stream(input_data, config=config):
            for node_name, state_update in step.items():
                elapsed = time.time() - graph_start
                detail = _extract_detail(node_name, state_update, elapsed=elapsed)
                step_id = f"{parent_id}_{step_counter}"

                history.append({
                    "role": "assistant", "content": detail,
                    "metadata": {
                        "title": NODE_STEP_MAP.get(node_name, {}).get("title", node_name),
                        "id": step_id, "parent_id": parent_id, "status": "done",
                    },
                })
                yield history

                step_counter += 1

                # Capture final assistant text
                if node_name in ("format_response", "guardian", "clarify"):
                    for msg in state_update.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response_text = msg.content.replace(settings.memory_tag, "").strip()
                break

        total_time = time.time() - graph_start

        # Mark accordions done
        for msg in history:
            meta = (msg or {}).get("metadata") or {}
            if meta.get("id") == parent_id or meta.get("parent_id") == parent_id:
                msg["metadata"]["status"] = "done"

        yield history

        # --- Stream final text ---
        history.append({"role": "assistant", "content": ""})
        for char in final_response_text:
            history[-1]["content"] += char
            time.sleep(0.005)
            yield history

        if not final_response_text and history[-1].get("content") == "":
            history.pop()

        # --- Append total response time ---
        total_display = _format_elapsed(total_time)
        history.append({
            "role": "assistant",
            "content": f"⏱️ **Response time: {total_display}**",
        })
        yield history

    except Exception as e:
        logger.error(f"AI Invocation Error: {e}", exc_info=True)
        for msg in history:
            meta = (msg or {}).get("metadata") or {}
            if meta.get("id") == parent_id:
                msg["metadata"]["status"] = "done"
                break
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        yield history

    # --- Save thread metadata ---
    try:
        save_thread_metadata(user_id, thread_id, CHATBOT_GRAPH, message)
    except Exception as e:
        logger.error(f"Failed to save thread metadata: {e}")


# ### --- UI HELPER FUNCTIONS --- ###


def add_message(message: str, history: List[Dict[str, Any]]) -> tuple:
    """Instantly adds user message to chat history (no blocking)."""
    if not message.strip():
        return "", history, None
    history.append({"role": "user", "content": message})
    return "", history, message


def refresh_threads(thread_id: str, user_id: str) -> list:
    """Returns updated thread table data for gr.Dataframe."""
    return get_threads_table(user_id, CHATBOT_GRAPH)


def refresh_chat(thread_id: str, user_id: str) -> List[Dict[str, str]]:
    """Loads chat history for the selected thread."""
    return load_chat_history(CHATBOT_GRAPH, thread_id, user_id)


def new_chat(user_id: str) -> tuple:
    """Creates a new chat thread."""
    new_id = str(uuid.uuid4())
    threads = get_threads_table(user_id, CHATBOT_GRAPH)
    return new_id, threads, []


def delete_thread(selected_row, thread_id: str, user_id: str) -> tuple:
    """Deletes the currently selected thread."""
    import pandas as pd
    if selected_row is not None and not (isinstance(selected_row, pd.DataFrame) and selected_row.empty):
        try:
            if isinstance(selected_row, pd.DataFrame):
                rows = selected_row.values.tolist()
                tid = rows[0][1] if rows else None
            elif isinstance(selected_row, (list, tuple)) and len(selected_row) > 1:
                tid = selected_row[1]
            else:
                tid = None

            if tid:
                from src.services.database import delete_thread_data
                delete_thread_data(user_id, tid, CHATBOT_GRAPH.store)
        except Exception as e:
            logger.error(f"Error deleting thread: {e}")

    new_id = str(uuid.uuid4())
    threads = get_threads_table(user_id, CHATBOT_GRAPH)
    return new_id, threads, []


def select_thread(evt: gr.SelectData) -> tuple:
    """Switches to the selected thread."""
    if evt and evt.value and len(evt.value) > 1:
        tid = evt.value[1]
        history = load_chat_history(CHATBOT_GRAPH, tid, settings.default_user_id)
        return tid, history
    return settings.default_user_id, []


# ### --- GRADIO BLOCKS APP --- ###


CUSTOM_CSS = """
/* Word-wrap SQL code blocks in chatbot output */
.chatbot .code_block_wrapper pre,
.chatbot pre {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    word-break: break-all !important;
    overflow-x: auto !important;
    max-width: 100% !important;
}
.chatbot code {
    white-space: pre-wrap !important;
    word-break: break-all !important;
}
.chatbot {
    overflow-x: hidden !important;
}
"""

with gr.Blocks(
    title="Personalized AI Assistant",
    fill_height=True,
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown("# 🧠 Personalized AI Assistant")
    gr.Markdown("*Persistent long-term memory via Neon DB & LangGraph Store.*")

    thread_id_state = gr.State(value=str(uuid.uuid4()))
    user_id_state = gr.State(value=settings.default_user_id)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 💬 Conversations")
            new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")

            thread_table = gr.Dataframe(
                headers=["Chat Name", "Thread ID", "Last Updated"],
                value=get_threads_table(settings.default_user_id, CHATBOT_GRAPH),
                interactive=False, wrap=True, max_height=250, show_search="search",
            )

            delete_btn = gr.Button("🗑️ Delete Selected Thread", variant="stop", size="sm")

            gr.Markdown("---")
            gr.Markdown("### 📖 Lessons")
            lessons_container = gr.HTML("")

            gr.Markdown("---")
            gr.Markdown(
                f"👤 User: **{settings.default_user_id}**  \n"
                f"Rate limit: **{settings.rate_limit_rpm} RPM**"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                render_markdown=True, line_breaks=True, autoscroll=False,
                buttons=["copy", "copy_all"], avatar_images=(None, None),
                placeholder="Start a conversation...", height=600,
                value=load_chat_history(
                    CHATBOT_GRAPH, settings.default_user_id, settings.default_user_id,
                ),
            )

            chat_input = gr.Textbox(
                placeholder="What's on your mind?",
                show_label=False, container=False,
            )

    # === EVENT WIRING ===
    message_state = gr.State(value="")

    chat_msg = chat_input.submit(
        add_message, [chat_input, chatbot], [chat_input, chatbot, message_state], queue=False
    )
    bot_msg = chat_msg.then(
        bot, [message_state, chatbot, thread_id_state, user_id_state], chatbot, queue=True
    )
    bot_msg.then(lambda: gr.Textbox(interactive=True), None, chat_input, queue=False)
    # Auto-refresh sidebar after response
    bot_msg.then(refresh_threads, [thread_id_state, user_id_state], thread_table, queue=False)

    new_chat_btn.click(new_chat, [user_id_state], [thread_id_state, thread_table, chatbot])

    delete_btn.click(
        delete_thread, [thread_table, thread_id_state, user_id_state],
        [thread_id_state, thread_table, chatbot]
    )

    thread_table.select(select_thread, None, [thread_id_state, chatbot])


# ### --- LAUNCH --- ###

if __name__ == "__main__":
    demo.launch(share=True)
