import threading

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.services.llm import get_llm
from src.services.sql_engine import sql_engine
from src.prompts.sql_agent import (
    get_sql_generation_prompt, 
    get_sql_healing_prompt, 
    get_sql_response_format_prompt
)
from src.prompts.assistant import get_assistant_prompt
from src.core.config import settings
from src.workflow.state import State
from src.workflow.schema import SQLResponse
from src.utils.logger import logger, log_context
from src.utils.table import generate_markdown_table
from src.utils.limiter import rate_limiter
from src.services.lessons import get_relevant_lessons, record_lesson
from src.workflow.schema import (
    SQLResponse, 
    GuardianOutput, 
    ClassifierOutput,
    LessonDistillationOutput, 
    SchemaSelectorOutput,
    AnchorSelection,
    ClarificationOutput,
    SQLGenerationOutput,
    ExecuteSQLOutput,
    ChatbotResponse
)

# Initialize single 8B model for ALL tasks
llm = get_llm()

def call_chatbot(state: State, config: RunnableConfig, store=None):
    """
    Standard chatbot node with Systemic Lessons.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info(f"Node: call_chatbot | User: {user_id}")
        last_user_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Recent messages + Sliding window of history
        window_size = getattr(settings, "context_window_size", 20)
        current_chat_history = state["messages"][-window_size:]
        
        # --- LEVEL 3: SYSTEMIC CONTEXT (Lessons from Mistakes) ---
        lessons_text, applied_titles = get_relevant_lessons(last_user_msg, store)

        # --- ASSEMBLE & INVOKE ---
        prompt_template = get_assistant_prompt()
        chain = prompt_template | llm.with_structured_output(ChatbotResponse)
        
        logger.info(f"Chatbot Node | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")
        
        res = chain.invoke({
            "lessons": lessons_text,
            "messages": current_chat_history
        })

        # Internal Validation -> Export to Dict
        res.node_name = "call_chatbot"
        return {"messages": [AIMessage(content=res.response)]}
    finally:
        log_context.reset(token)

def guardian_node(state: State, config: RunnableConfig, store=None):
    """Entry Point: Categorizes intent as SQL, DENY, or CLARIFY. Supports Stateful Context Lock."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info("Node: guardian_node")
        
        # 1. Global Rate Limit Check
        if not rate_limiter.check_and_record():
            logger.warning("Global Rate Limit Reached!")
            return {
                "intent": "DENY",
                "messages": [AIMessage(content="⚠️ System busy. Please wait a moment.")]
            }
        
        last_msg = state["messages"][-1].content
        is_locked = state.get("is_awaiting_clarification", False)
        vague_context = state.get("vague_query_context", "")

        domain_summary = """
        This database (Pagila) contains DVD rental business data:
        1. PEOPLE: Actors, Customers, Staff members.
        2. INVENTORY: Films, Categories, Languages, Inventories, Stores.
        3. BUSINESS: Rentals, Payments, Addresses, Cities, Countries.
        """

        if is_locked:
            decision_prompt = f"""You are a Context-Aware Security Guardian.
The user previously made a vague request: "{vague_context}".
We asked for clarification. Their latest message is: "{last_msg}".

DOMAIN SUMMARY:
{domain_summary}

### CLASSIFICATION RULES:
- SQL: The new message, combined with the previous context, now forms a clear and valid SQL request about the domain.
- CLARIFY: The message is related but still vague. We stay in the clarification loop.
- DENY: The user has clearly changed the subject to something irrelevant OR is asking to modify data.

Note: If the user provides a COMPLETELY NEW but valid SQL request (e.g., pivoting from 'films' to 'customers'), classify as SQL and we will switch context.
"""
        else:
            decision_prompt = f"""You are an Intent Classifier and Security Guardian.
Analyze the user message and determine if it relates to the database domain described below.

DOMAIN SUMMARY:
{domain_summary}

### CLASSIFICATION RULES:
- SQL: The request is a clear, actionable question about querying or analyzing data from the domain.
- CLARIFY: The request is related to the domain but is too vague, ambiguous, or underspecified to generate SQL for (e.g., "Tell me about films" without criteria).
- DENY: The request is NOT about the domain, asks to MODIFY data, or is general chitchat.

User Message: "{last_msg}"
"""
        # Use structured output for determinism
        chain = llm.with_structured_output(GuardianOutput)
        res = chain.invoke([SystemMessage(content=decision_prompt)])
        res.node_name = "guardian"
        
        logger.info(f"Guardian Action: {res.intent} | Context Locked: {is_locked} | Thought: {res.thought_process}")

        # Update logs for observability
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())

        if res.intent == "DENY":
            # If we were locked but they pivot to something invalid, reset the lock
            return {
                "intent": "DENY",
                "is_awaiting_clarification": False,
                "vague_query_context": "",
                "agent_logs": logs,
                "messages": [AIMessage(content="I specialize exclusively in the Pagila DVD Rental database. I cannot assist with personal requests, general chat, or data modification.")]
            }
        
        if res.intent == "CLARIFY":
            return {
                "intent": "CLARIFY",
                "is_awaiting_clarification": True,
                "vague_query_context": last_msg if not is_locked else f"{vague_context} + {last_msg}",
                "agent_logs": logs
            }
        
        # If SQL intent is reached, reset the clarification lock
        return {
            "intent": "SQL", 
            "is_awaiting_clarification": False,
            "vague_query_context": "",
            "agent_logs": logs
        }
    finally:
        log_context.reset(token)

def classifier_node(state: State, config: RunnableConfig, store=None):
    """Determines if the SQL query is SIMPLE (one table) or COMPLEX (joins/logic)."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})       
    
    try:
        logger.info("Node: classifier_node")
        last_msg = state["messages"][-1].content
        
        prompt = f"""You are a SQL Query Planner. 
Analyze the user question and determine if it requires joining multiple tables.

User Question: "{last_msg}"

### GUIDELINES:
- SIMPLE: Questions that can be answered using a SINGLE table (e.g., "List films", "Count customers", "Top 10 most expensive films"). 
- COMPLEX: Questions requiring JOINS between 2 or more tables (e.g., "Which customers rented Action films?", "Revenue by category").

NOTE: Sorting or limiting on a single table is STILL SIMPLE.
"""
        chain = llm.with_structured_output(ClassifierOutput)
        res = chain.invoke([SystemMessage(content=prompt)])
        res.node_name = "classifier"
        
        logger.info(f"Classification: {'COMPLEX' if res.is_complex else 'SIMPLE'} | Thought: {res.thought_process}")
        
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())

        return {"is_complex": res.is_complex, "agent_logs": logs}
    finally:
        log_context.reset(token)

def clarify_node(state: State, config: RunnableConfig):
    """Asks the user for clarification when the intent is ambiguous."""
    logger.info("Node: clarify_node")
    
    # Use structured output
    chain = llm.with_structured_output(ClarificationOutput)
    prompt = f"The user asked: '{state['messages'][-1].content}'. This is too vague for a SQL query. Ask a concise follow-up question to clarify what they want to see from the DVD rental database."
    res = chain.invoke(prompt)
    
    return {"messages": [AIMessage(content=res.clarification_question)]}

def anchor_selector_node(state: State, config: RunnableConfig, store=None):
    """Hybrid Discovery Phase 1: Two-Pass Entity Extraction & Physical Table Mapping."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})       

    try:
        logger.info("Node: anchor_selector")
        last_msg = state["messages"][-1].content
        all_tables = sql_engine.list_tables()

        # Pass 1: Semantic Entity Extraction (Structured)
        entity_prompt = f"""Identify the core database entities and filters mentioned in this question.        
Question: "{last_msg}"
Example Entities: 'Canada', 'Action', 'most rentals', 'spent'.
"""
        entity_chain = llm.with_structured_output(AnchorSelection)
        entity_res = entity_chain.invoke([SystemMessage(content=entity_prompt)])
        entities = ", ".join(entity_res.anchors)

        # Pass 2: Hard Physical Table Mapping
        mapping_prompt = f"""You are a Database Architect. Map the following entities to the specific PHYSICAL tables needed to query them.
Entities Found: {entities}
Available Tables: {all_tables}

### CRITICAL RULES:
- If 'Canada' or geographic filters are mentioned, include 'country'.
- If 'Action' or categories are mentioned, include 'category'.
- If 'spent' or 'amount' is mentioned, include 'payment'.
- NEVER select views (ending in '_info' or '_list').
"""
        anchor_chain = llm.with_structured_output(AnchorSelection)
        anchor_res = anchor_chain.invoke([SystemMessage(content=mapping_prompt)])
        anchors = [a for a in anchor_res.anchors if a in all_tables]

        # 3. Deterministic FK Bridge Traversal
        bridges = sql_engine.get_bridge_tables(anchors)
        selected_tables = list(set(anchors + bridges))

        logger.info(f"Join Topology: Anchors={anchors} | Bridges={bridges}")

        logs = state.get("agent_logs", [])
        logs.append({
            "node_name": "anchor_selector",
            "anchors": anchors,
            "bridges": bridges,
            "selected_tables": selected_tables,
            "thought_process": getattr(anchor_res, "thought_process", "")
        })

        return {"selected_tables": selected_tables, "agent_logs": logs}
    finally:
        log_context.reset(token)

def column_pruner_node(state: State, config: RunnableConfig, store=None):
    """Hybrid Discovery Phase 2: Surgically prunes columns while protecting Join Keys."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})       
    
    try:
        logger.info("Node: column_pruner")
        last_msg = state["messages"][-1].content
        selected_tables = state["selected_tables"]
        
        # Fetch relationships and partial schema
        fk_relationships = sql_engine.get_relevant_fks(selected_tables)
        partial_schema = sql_engine.get_schema(selected_tables)
        
        pruning_prompt = f"""You are a Data Architect. Prune the schema below to ONLY include the columns needed for this question.
Question: "{last_msg}"
Schema:
{partial_schema}
Relationships:
{fk_relationships}

### CRITICAL RULES:
1. You MUST retain ALL columns mentioned in the 'Relationships' list (Join Keys).
2. Retain columns needed for filters (WHERE), ordering (ORDER BY), and display (SELECT).
"""
        chain = llm.with_structured_output(SchemaSelectorOutput)
        res = chain.invoke([SystemMessage(content=pruning_prompt)])
        res.node_name = "column_pruner"
        
        # Deterministic Guard: Ensure all selected tables exist and Join Keys are preserved
        pruned_cols = res.selected_columns
        for table in selected_tables:
            if table not in pruned_cols: pruned_cols[table] = []
            
        for rel in fk_relationships:
            if rel["source_column"] not in pruned_cols[rel["source_table"]]:
                pruned_cols[rel["source_table"]].append(rel["source_column"])
            if rel["target_column"] not in pruned_cols[rel["target_table"]]:
                pruned_cols[rel["target_table"]].append(rel["target_column"])
        
        logger.info(f"Pruning complete for {len(pruned_cols)} tables.")
        
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())
        
        return {
            "selected_columns": pruned_cols, 
            "fk_relationships": fk_relationships,
            "agent_logs": logs
        }
    finally:
        log_context.reset(token)
def generate_sql_node(state: State, config: RunnableConfig, store=None):
    """Generates SQL query using surgically pruned schema and tiered lessons."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})       
    
    try:
        logger.info("Node: generate_sql")
        user_question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        
        # Priority 1: Use surgical pruned columns from state
        # Priority 2: Use selected tables from state
        # Priority 3: Fallback to full schema (Simple queries)
        
        selected_columns = state.get("selected_columns")
        selected_tables = state.get("selected_tables")
        
        if selected_columns:
            logger.info(f"Using SURGICALLY PRUNED schema ({len(selected_columns)} tables)")
            schema_str = str(selected_columns)
        elif selected_tables:
            logger.info(f"Using table-filtered schema ({len(selected_tables)} tables)")
            full_schema = sql_engine.get_schema_object()
            schema_str = str({t: full_schema.get(t, []) for t in selected_tables})
        else:
            logger.info("Using full schema fallback")
            schema_str = str(sql_engine.get_schema_object())
        
        # Tiered Lesson Retrieval
        lessons_text, applied_titles = get_relevant_lessons(
            user_question, 
            store, 
            selected_tables=selected_tables
        )
        
        prompt_template = get_sql_generation_prompt()
        chain = prompt_template | llm.with_structured_output(SQLGenerationOutput)
        
        res = chain.invoke({
            "schema": schema_str,
            "lessons": lessons_text,
            "history": state["messages"][:-1],
            "question": user_question
        })
        
        sql_query = res.sql.strip().replace("```sql", "").replace("```", "")
        return {"current_sql": sql_query, "retry_count": 0}
    finally:
        log_context.reset(token)

def execute_sql_node(state: State, config: RunnableConfig):
    """
    Executes the current SQL query and captures results or errors.
    Detects if the result is an aggregate (1x1).
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info(f"Node: execute_sql | Query: {state['current_sql']}")
        raw_result = sql_engine.execute_query(state["current_sql"])
        
        # Use Pydantic for internal normalization
        result = ExecuteSQLOutput(**raw_result)

        if result.status == "success":
            data = result.data
            # Detect 1x1 shape (Aggregate)
            is_aggregated = False
            if len(data) == 1:
                row = data[0]
                if len(row.keys()) == 1:
                    is_aggregated = True
            
            logger.info(f"Execution Success. Rows: {result.row_count} | Aggregated: {is_aggregated}")
            return {"sql_results": data, "is_aggregated": is_aggregated, "sql_error": None}
        else:
            logger.warning(f"Execution Failed. Error: {result.error_message}")
            return {"sql_error": result.error_message, "sql_results": [], "is_aggregated": False}
    finally:
        log_context.reset(token)

def _background_distill_lesson(state_data: dict, store, user_id: str):
    """
    Background worker to distill and record high-quality lessons using nested Pydantic structure.
    """
    try:
        current_sql = state_data['current_sql']
        sql_error = state_data['sql_error']
        fixed_sql = state_data['fixed_sql']
        selected_tables = state_data.get('selected_tables')

        learning_prompt = f"""You are a Senior Staff Engineer mentoring junior agents.
Create a "Golden standard Lesson" from this SQL mistake using the nested structure.

### CONTEXT:
- FAILED QUERY: {current_sql}
- ERROR REPORTED: {sql_error}
- CORRECTED FIX: {fixed_sql}
- TABLES INVOLVED: {selected_tables or 'unknown'}

### INSTRUCTIONS:
1. `instruction`: A single, clear, actionable rule.
2. `mistake`: Describe exactly what went wrong.
3. `reasoning`: Provide a Markdown report including:
   - **Root Cause Analysis**
   - **Future Proofing**
4. `example`: Populate with the original error and the fixed SQL.
5. `ending_note`: MUST be "By following this instruction, future models can avoid this specific mistake and write more robust and error-free SQL queries."
"""
        distiller = llm.with_structured_output(LessonDistillationOutput)
        lesson = distiller.invoke([SystemMessage(content=learning_prompt)])
        
        # Construct the "Neat and Beautiful" Markdown reasoning for the database
        formatted_reasoning = f"""### Mistake
{lesson.body.mistake}

{lesson.body.reasoning}

### Example Comparison
- **Original Error:**
```sql
{lesson.body.example.original_error}
```
- **Fixed SQL:**
```sql
{lesson.body.example.fixed_sql}
```

---
*{lesson.ending_note}*"""

        record_lesson(
            lesson.title, 
            sql_error, 
            lesson.body.instruction, 
            formatted_reasoning, 
            store, 
            is_global=lesson.is_global,
            tags=selected_tables
        )
        logger.info(f"[{user_id}] | SUCCESS | Background Task: Recorded lesson: {lesson.title}")
        
    except Exception as e:
        logger.error(f"Background lesson distillation failed: {e}", exc_info=True)

def heal_sql_node(state: State, config: RunnableConfig, store=None):
    """Heals SQL and offloads lesson distillation to a background task."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})       
    
    try:
        retry = state.get("retry_count", 0) + 1
        logger.info(f"Node: heal_sql | Attempt: {retry}")
        
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        selected_tables = state.get("selected_tables")
        
        # Fetch schema from engine (cached)
        full_schema = sql_engine.get_schema_object()
        if selected_tables:
            filtered_schema = {t: full_schema.get(t, []) for t in selected_tables}
            schema_str = str(filtered_schema)
        else:
            schema_str = str(full_schema)
        
        prompt_template = get_sql_healing_prompt()
        # Use structured output for robust SQL extraction
        chain = prompt_template | llm.with_structured_output(SQLGenerationOutput)
        res = chain.invoke({
            "schema": schema_str,
            "failed_query": state["current_sql"],
            "error_message": state["sql_error"],
            "question": user_question
        })
        
        fixed_sql = res.sql.strip().replace("```sql", "").replace("```", "")
        
        # --- OFFLOAD TO BACKGROUND ---
        if store:
            state_data = {
                "current_sql": state["current_sql"],
                "sql_error": state["sql_error"],
                "fixed_sql": fixed_sql,
                "selected_tables": selected_tables,
                "user_question": user_question
            }
            threading.Thread(
                target=_background_distill_lesson,
                args=(state_data, store, user_id),
                daemon=True
            ).start()
            logger.info("Lesson distillation offloaded to background thread.")

        return {"current_sql": fixed_sql, "retry_count": retry, "sql_error": None}
    finally:
        log_context.reset(token)


def format_sql_response_node(state: State, config: RunnableConfig):
    """
    Ultra-low latency renderer using Flash (8B) for summaries and Python for tables.
    Prunes input data to 5 rows and handles 1x1 aggregates surgically.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info(f"Node: format_sql_response | Aggregated Flag: {state.get('is_aggregated')}")
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        raw_results = state.get("sql_results", [])
        is_aggregated = state.get("is_aggregated", False)

        # 1. Prune Data for LLM (First 5 rows only)
        sample_data = raw_results[:5] if len(raw_results) > 5 else raw_results
        
        # 2. Invoke Flash Model for Summary
        prompt_template = get_sql_response_format_prompt()
        # Explicitly use llm (Flash 8B) for speed
        chain = prompt_template | llm.with_structured_output(SQLResponse)
        
        try:
            response = chain.invoke({
                "question": user_question,
                "query": state["current_sql"],
                "data": str(sample_data)
            })
            
            output_parts = []
            
            # A. Summary / Natural Language Answer
            if response.summary:
                output_parts.append(response.summary)
            elif is_aggregated and raw_results:
                # Fallback for aggregates if LLM fails to provide a summary
                val = list(raw_results[0].values())[0]
                output_parts.append(f"The result is {val}.")

            # B. Table (Only for non-aggregated lists)
            if not is_aggregated and raw_results:
                table_md = generate_markdown_table(raw_results)
                output_parts.append(table_md)
                
            # C. SQL Block
            if state.get("current_sql"):
                sql_block = f"**Executed SQL:**\n```sql\n{state['current_sql'].strip()}\n```"
                output_parts.append(sql_block)
                
            final_content = "\n\n".join(output_parts)
            
            if not final_content:
                final_content = "I found no results for your query in the database."
                
            return {"messages": [AIMessage(content=final_content)]}
            
        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            return {"messages": [AIMessage(content="I encountered an error while formatting the data.")]}
    finally:
        log_context.reset(token)
