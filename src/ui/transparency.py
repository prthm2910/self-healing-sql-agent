import streamlit as st
from typing import List, Dict, Any
from src.utils.limiter import rate_limiter
from src.utils.key_manager import groq_key_manager

def render_transparency_log(agent_logs: List[Dict[str, Any]] = None):
    """
    Renders the Developer Console / Transparency Log.
    Provides visibility into the agent's internal reasoning, API status, and Data Diet.
    """
    st.divider()
    with st.expander("🛠️ Developer Console / Transparency Log", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📊 API Status", "🧠 Thought Trace", "🥗 Data Diet"])

        with tab1:
            _render_api_stats()

        with tab2:
            _render_thought_trace(agent_logs)

        with tab3:
            _render_data_diet(agent_logs)

def _render_api_stats():
    """Shows real-time token usage and key rotation status."""
    limiter_stats = rate_limiter.get_stats()
    key_stats = groq_key_manager.get_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active API Keys", f"{key_stats['active_keys']}/{key_stats['total_keys']}")
    with col2:
        st.metric("Requests/Min (RPM)", f"{limiter_stats['rpm']}/{limiter_stats['rpm_limit']}")
    with col3:
        # Calculate percentage of TPM used
        tpm_percent = (limiter_stats['tpm'] / limiter_stats['tpm_limit']) * 100 if limiter_stats['tpm_limit'] > 0 else 0
        st.metric("Tokens/Min (TPM)", f"{limiter_stats['tpm']:,}", f"{tpm_percent:.1f}% limit")

    if key_stats['blacklisted_keys'] > 0:
        st.warning(f"⚠️ {key_stats['blacklisted_keys']} key(s) currently in 24h cooldown due to daily limits.")

def _render_thought_trace(agent_logs: List[Dict[str, Any]]):
    """Displays the execution path and internal monologue of each node."""
    if not agent_logs:
        st.info("No logs available for this turn. Start a conversation to see the trace.")
        return

    for i, entry in enumerate(agent_logs):
        node_name = entry.get("node_name", f"Step {i+1}").upper()
        thought = entry.get("thought_process", "No thought recorded.")
        
        with st.container():
            st.markdown(f"**Node:** `{node_name}`")
            st.info(thought)
            
            # Show raw JSON details in a nested expander for deep debugging
            with st.expander("Raw Metadata", expanded=False):
                st.json(entry)
            st.divider()

def _render_data_diet(agent_logs: List[Dict[str, Any]]):
    """Visualizes the schema pruning results."""
    if not agent_logs:
        st.info("No discovery data available.")
        return

    # Find the discovery node logs
    discovery_log = next((l for l in agent_logs if l.get("node_name") == "anchor_selector"), None)
    pruning_log = next((l for l in agent_logs if l.get("node_name") == "column_pruner"), None)

    if discovery_log:
        st.markdown("### 🔍 Vector-Selected Tables (Anchors)")
        tables = discovery_log.get("selected_tables", [])
        if tables:
            # Create a nice visual pill-based list
            cols = st.columns(len(tables) if len(tables) < 5 else 5)
            for j, table in enumerate(tables):
                cols[j % 5].code(table)
        else:
            st.write("No tables selected.")

    if pruning_log:
        st.markdown("### ✂️ Column Pruning Results")
        selected_columns = pruning_log.get("selected_columns", {})
        if selected_columns:
            for table, cols in selected_columns.items():
                with st.expander(f"Table: {table}", expanded=False):
                    st.write(", ".join(cols))
        else:
            st.write("No columns pruned.")
