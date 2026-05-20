import threading
import time
import json
import sqlglot
from typing import Dict, Any, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.services.llm import get_llm
from src.services.sql_engine import sql_engine
from src.services.sql_transpiler import SQLTranspiler
from src.prompts.sql_agent import (
    get_sql_generation_prompt, 
    get_sql_healing_prompt,     
    get_sql_response_format_prompt,
    get_decomposer_prompt,
    get_worker_prompt
)
from src.prompts.assistant import get_assistant_prompt
from src.core.config import settings
from src.workflow.state import State
from src.utils.logger import logger, log_context
from src.utils.table import generate_markdown_table
from src.utils.limiter import rate_limiter
from src.services.lessons import get_relevant_lessons, record_lesson
from src.workflow.schema import (
    SQLResponse, 
    GuardianLLMOutput, GuardianNodeOutput,
    ClassifierLLMOutput, ClassifierNodeOutput,
    LessonDistillationOutput, 
    SchemaSelectorLLMOutput, SchemaSelectorNodeOutput,
    QueryBlueprint, SQLResultSet,
    DecomposerOutput, SubTask, JoinPlan,
    AnchorSelection, ClarificationOutput,
    SQLSelection, SQLFilter
)

# Initialize single 8B model for ALL tasks
llm = get_llm()

def invoke_with_fallback(prompt: Any, output_model: type, inputs: dict = None) -> Any:
    """
    Robustly invokes the LLM with structured output, falling back to manual 
    parsing if JSON mode fails (common with Groq's strictness).
    """
    structured_llm = llm.with_structured_output(output_model)

    if hasattr(prompt, "invoke"):
        chain = prompt | structured_llm
        to_invoke = inputs or {}
    else:
        chain = structured_llm
        to_invoke = prompt

    try:
        return chain.invoke(to_invoke)
    except Exception as e:
        logger.warning(f"Structured output failed for {output_model.__name__}: {e}. Attempting robust fallback...")

        # Fallback: Manual Parsing
        if hasattr(prompt, "format") and inputs:
            raw_res = llm.invoke(prompt.format(**inputs))
        elif hasattr(prompt, "invoke") and inputs:
            raw_res = llm.invoke(prompt.invoke(inputs))
        else:
            raw_res = llm.invoke(prompt)

        content = raw_res.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(content)
            return output_model.model_validate(parsed)
        except Exception as parse_err:
            logger.error(f"Fallback parsing failed: {parse_err}")
            raise e

# --- CONVERSATIONAL & INTENT NODES ---

def call_chatbot(state: State, config: RunnableConfig, store=None):
    """Chatbot node with Memory + Lessons."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})
    try:
        lessons_text, applied_titles = get_relevant_lessons(state["messages"][-1].content, store)
        prompt_template = get_assistant_prompt()
        res = (prompt_template | llm).invoke({
            "lessons": lessons_text,
            "messages": state["messages"][-10:]
        })
        return {"messages": [AIMessage(content=res.content)]}
    finally:
        log_context.reset(token)

def guardian_node(state: State, config: RunnableConfig, store=None):
    """Categorizes intent as SQL, DENY, or CLARIFY."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})
    
    try:
        logger.info("Node: guardian_node")
        if not rate_limiter.check_and_record():
            return {"intent": "DENY", "messages": [AIMessage(content="⚠️ System busy.")]}
        
        last_msg = state["messages"][-1].content
        is_locked = state.get("is_awaiting_clarification", False)
        vague_context = state.get("vague_query_context", "")

        prompt = f"Analyze intent for: '{last_msg}'. (Prev: '{vague_context}' if locked). Output: SQL, CLARIFY, or DENY."
        llm_res = invoke_with_fallback([SystemMessage(content=prompt)], GuardianLLMOutput)
        llm_res.node_name = "guardian"

        logs = state.get("agent_logs", [])
        logs.append(llm_res.model_dump())

        if llm_res.intent == "DENY":
            return {"intent": "DENY", "is_awaiting_clarification": False, "messages": [AIMessage(content="I specialize in the Pagila DB. I can't assist.")], "agent_logs": logs}
        
        if llm_res.intent == "CLARIFY":
            return {"intent": "CLARIFY", "is_awaiting_clarification": True, "vague_query_context": last_msg, "agent_logs": logs}
        
        return {"intent": "SQL", "db_schema": sql_engine.get_schema_object(), "is_awaiting_clarification": False, "agent_logs": logs}
    finally:
        log_context.reset(token)

def classifier_node(state: State, config: RunnableConfig, store=None):
    """SIMPLE vs COMPLEX classification."""
    token = log_context.set({"user_id": config.get("configurable", {}).get("user_id"), "thread_id": config.get("configurable", {}).get("thread_id")})
    try:
        prompt = f"SIMPLE (1 table) or COMPLEX (joins) for: '{state['messages'][-1].content}'?"
        llm_res = invoke_with_fallback([SystemMessage(content=prompt)], ClassifierLLMOutput)
        llm_res.node_name = "classifier"
        logs = state.get("agent_logs", [])
        logs.append(llm_res.model_dump())
        return {"is_complex": llm_res.is_complex, "agent_logs": logs}
    finally:
        log_context.reset(token)

def clarify_node(state: State, config: RunnableConfig):
    """Asks for clarification."""
    prompt = f"The user asked: '{state['messages'][-1].content}'. Ask a follow-up to clarify for SQL generation."
    res = invoke_with_fallback(prompt, ClarificationOutput)
    return {"messages": [AIMessage(content=res.clarification_question)]}

# --- DISCOVERY & PLANNING NODES ---

def anchor_selector_node(state: State, config: RunnableConfig, store=None):
    """Discovery Phase 1: Entity Mapping."""
    token = log_context.set({"user_id": config.get("configurable", {}).get("user_id"), "thread_id": config.get("configurable", {}).get("thread_id")})
    try:
        all_tables = sql_engine.list_tables()
        prompt = f"""You are a Database Architect. Identify ALL relevant 'Anchor' tables mentioned or implied.
Question: "{state['messages'][-1].content}"
Available Tables: {all_tables}
Return a comma-separated list of physical table names ONLY."""
        res = invoke_with_fallback([SystemMessage(content=prompt)], AnchorSelection)
        anchors = [a for a in res.anchors if a in all_tables]
        bridges = sql_engine.get_bridge_tables(anchors)
        selected_tables = list(set(anchors + bridges))
        
        logs = state.get("agent_logs", [])
        logs.append({"node_name": "anchor_selector", "selected_tables": selected_tables, "thought_process": res.thought_process})
        return {"selected_tables": selected_tables, "agent_logs": logs}
    finally:
        log_context.reset(token)

def column_pruner_node(state: State, config: RunnableConfig, store=None):
    """Discovery Phase 2: Column Pruning."""
    token = log_context.set({"user_id": config.get("configurable", {}).get("user_id"), "thread_id": config.get("configurable", {}).get("thread_id")})
    try:
        selected_tables = state["selected_tables"]
        fk_relationships = sql_engine.get_relevant_fks(selected_tables)
        partial_schema = sql_engine.get_schema(selected_tables)
        prompt = f"Prune columns for: '{state['messages'][-1].content}'. Schema: {partial_schema}. FKs: {fk_relationships}."
        llm_res = invoke_with_fallback([SystemMessage(content=prompt)], SchemaSelectorLLMOutput)
        llm_res.node_name = "column_pruner"
        
        # Mapping logic (Legacy compat)
        pruned_cols = {item.table_name: item.columns for item in llm_res.selected_columns}
        for rel in fk_relationships:
            pruned_cols.setdefault(rel["source_table"], []).append(rel["source_column"])
            pruned_cols.setdefault(rel["target_table"], []).append(rel["target_column"])
            
        logs = state.get("agent_logs", [])
        logs.append(llm_res.model_dump())
        return {"selected_columns": pruned_cols, "agent_logs": logs}
    finally:
        log_context.reset(token)

# --- EXECUTION NODES ---

def decomposer_node(state: State, config: RunnableConfig):
    """Manager: Divide & Conquer planning with Deterministic Join Key Injection."""
    token = log_context.set({"user_id": config.get("configurable", {}).get("user_id"), "thread_id": config.get("configurable", {}).get("thread_id")})
    try:
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        full_schema = sql_engine.get_schema_object(state.get("selected_tables"))
        res = invoke_with_fallback(get_decomposer_prompt(), DecomposerOutput, {"skeleton_schema": str(full_schema), "question": user_question})
        res.node_name = "decomposer"

        # --- DETERMINISTIC JOIN KEY INJECTION ---
        # Ensure join keys are present in Worker sub-task required_columns
        task_map = {t.task_id: t for t in res.sub_tasks}
        for step in res.join_plan.steps:
            for tid in [step.left, step.right]:
                if tid in task_map:
                    if step.on not in task_map[tid].required_columns:
                        logger.info(f"Injecting missing join key '{step.on}' into {tid}")
                        task_map[tid].required_columns.append(step.on)

        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())
        return {
            "sub_tasks": [t.model_dump() for t in res.sub_tasks],
            "join_plan": res.join_plan.model_dump(),
            "sql_snippets": {},
            "current_task_index": 0,
            "agent_logs": logs
        }
    finally:
        log_context.reset(token)

def generate_sql_node(state: State, config: RunnableConfig, store=None):
    """SIMPLE Query Generator (Blueprint)."""
    token = log_context.set({"user_id": config.get("configurable", {}).get("user_id"), "thread_id": config.get("configurable", {}).get("thread_id")})
    try:
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        schema_str = str(state.get("selected_columns") or sql_engine.get_schema_object(state.get("selected_tables")))
        lessons_text, _ = get_relevant_lessons(user_question, store, selected_tables=state.get("selected_tables"))
        
        blueprint = invoke_with_fallback(get_sql_generation_prompt(), QueryBlueprint, {
            "schema": schema_str, "lessons": lessons_text, "history": state["messages"][:-1], "question": user_question
        })
        blueprint.node_name = "sql_generator"
        sql_query = SQLTranspiler.to_sql(blueprint)
        
        logs = state.get("agent_logs", [])
        logs.append(blueprint.model_dump())
        return {"current_sql": sql_query, "retry_count": 0, "agent_logs": logs}
    finally:
        log_context.reset(token)

def worker_node(state: State, config: RunnableConfig):
    """Worker: Snippet Generator."""
    task = state["sub_tasks"][state.get("current_task_index", 0)]
    partial_schema = sql_engine.get_schema(task["tables"])
    prompt = f"""You are a Reliable SQL Worker. Solve sub-task: '{task['description']}'.
REQUIRED SELECT COLUMNS (for joining): {task['required_columns']}
SCHEMA: {partial_schema}
Output ONLY valid PostgreSQL SQL."""
    res = llm.invoke(prompt)
    sql = res.content.strip().replace("```sql", "").replace("```", "").rstrip(";")
    sqlglot.parse_one(sql, read="postgres") # Validation
    snippets = state.get("sql_snippets", {})
    snippets[task["task_id"]] = sql
    return {"sql_snippets": snippets}

def assembler_node(state: State):
    """Assembler: CTE Stichting."""
    try:
        final_sql = SQLTranspiler.merge_snippets(state["sql_snippets"], state["join_plan"])
        return {"current_sql": final_sql}
    except Exception as e:
        return {"sql_error": f"Assembly Error: {str(e)}"}

def execute_sql_node(state: State, config: RunnableConfig):
    """SQL Execution."""
    try:
        raw_res = sql_engine.execute_query(state["current_sql"])
        result = SQLResultSet(**raw_res)
        if result.status == "success":
            is_aggregated = result.row_count == 1 and len(result.columns) == 1
            return {"sql_results": result.rows, "is_aggregated": is_aggregated, "sql_error": None}
        return {"sql_error": result.error_message, "sql_results": [], "is_aggregated": False}
    except Exception as e:
        return {"sql_error": str(e)}

def heal_sql_node(state: State, config: RunnableConfig, store=None):
    """Self-Healing via Blueprint."""
    user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    blueprint = invoke_with_fallback(get_sql_healing_prompt(), QueryBlueprint, {
        "schema": str(sql_engine.get_schema_object()),
        "failed_query": state["current_sql"],
        "error_message": state["sql_error"],
        "question": user_question
    })
    fixed_sql = SQLTranspiler.to_sql(blueprint)
    if store:
        threading.Thread(target=_background_distill_lesson, args=(state["sql_error"], fixed_sql, state.get("selected_tables"), store), daemon=True).start()
    return {"current_sql": fixed_sql, "retry_count": state.get("retry_count", 0) + 1, "sql_error": None}

def format_sql_response_node(state: State, config: RunnableConfig):
    """Renderer."""
    user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    sample_data = state.get("sql_results", [])[:5]
    response = invoke_with_fallback(get_sql_response_format_prompt(), SQLResponse, {
        "question": user_question, "query": state["current_sql"], "data": str(sample_data)
    })
    output = [response.summary]
    if not state.get("is_aggregated") and state.get("sql_results"):
        output.append(generate_markdown_table(state["sql_results"]))
    output.append(f"**Executed SQL:**\n```sql\n{state['current_sql'].strip()}\n```")
    return {"messages": [AIMessage(content="\n\n".join(output))]}

def _background_distill_lesson(error: str, fixed_sql: str, tables: list, store):
    """Background lesson distillation."""
    try:
        prompt = f"Distill lesson. Error: {error}. Fix: {fixed_sql}."
        lesson = invoke_with_fallback([SystemMessage(content=prompt)], LessonDistillationOutput)
        record_lesson(lesson.title, error, lesson.body.instruction, lesson.body.reasoning, store, is_global=lesson.is_global, tags=tables)
    except Exception as e:
        logger.error(f"Background lesson failed: {e}")
