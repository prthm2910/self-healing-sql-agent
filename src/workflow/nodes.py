import threading
import time
import sqlglot

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.services.llm import get_llm
from src.services.sql_engine import SQLTranspiler, sql_engine
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
    GuardianOutput, 
    ClassifierOutput,
    LessonDistillationOutput, 
    SchemaSelectorOutput,    
    AnchorSelection,
    ClarificationOutput,
    SQLGenerationOutput,
    ExecuteSQLOutput,
    ChatbotResponse,
    DecomposerOutput,
    WorkerOutput,
    SubTask,
    JoinPlan
)

# Initialize single 8B model for ALL tasks
llm = get_llm()

def robust_invoke(chain, input, schema_class, max_retries=2):
    """
    Invokes a structured output chain with a fallback to manual JSON parsing if Groq's 
    'Tool choice is required' error occurs.
    """
    import json
    import re
    from langchain_core.runnables import RunnableSequence
    
    # 1. Try standard structured output
    try:
        return chain.invoke(input)
    except Exception as e:
        error_str = str(e).lower()
        # Log the specific failure for transparency
        if "400" in error_str and ("tool choice" in error_str or "tool_use_failed" in error_str):
            logger.warning(f"Structured output failed for {schema_class.__name__} (Attempt 1). Attempting robust fallback...")
        else:
            raise e # Not a tool failure, propagate original error

    # 2. Manual Fallback: Ask for raw JSON and parse it
    # We use the underlying LLM to avoid tool_choice=required
    raw_llm = llm # The global LoggedChatGroq instance
    
    # Resolve the prompt text from the chain and input
    prompt_text = ""
    try:
        if isinstance(chain, RunnableSequence):
            # Attempt to get the prompt by running the first part of the chain (the template)
            prompt_val = chain.first.invoke(input)
            prompt_text = prompt_val.to_string() if hasattr(prompt_val, "to_string") else str(prompt_val)
        elif isinstance(input, list):
            prompt_text = "\n".join([m.content for m in input if hasattr(m, 'content')])
        else:
            prompt_text = str(input)
    except Exception as p_err:
        logger.error(f"Failed to resolve prompt text for fallback: {p_err}")
        prompt_text = str(input)
        
    fallback_prompt = f"""{prompt_text}

### OUTPUT INSTRUCTIONS:
You MUST output ONLY a valid JSON object matching this schema:
{schema_class.model_json_schema()}

Ensure the output is a single valid JSON block enclosed in ```json ... ```.
"""
    
    for attempt in range(max_retries):
        try:
            res = raw_llm.invoke(fallback_prompt)
            content = res.content
            
            # Extract JSON from code blocks
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content
            
            # Remove any trailing commas or markdown artifacts
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Parse and validate with Pydantic
            parsed = json.loads(json_str)
            return schema_class(**parsed)
        except Exception as retry_err:
            logger.error(f"Fallback attempt {attempt+1} failed for {schema_class.__name__}: {retry_err}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to get valid structured output for {schema_class.__name__} after fallback attempts.") from retry_err
            time.sleep(1)

def decomposer_node(state: State, config: RunnableConfig):
    """
    The Manager Node: Decomposes complex queries into atomic sub-tasks and a Join Plan.
    Utilizes 'Skeleton Knowledge' (Tables + FK Paths) from the Anchor Selector.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    token = log_context.set({"user_id": user_id, "thread_id": configurable.get("thread_id", "unknown")})
    
    try:
        logger.info("Node: decomposer_node (Manager)")
        user_question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        
        selected_tables = state.get("selected_tables", [])
        fk_relationships = state.get("fk_relationships", [])
        
        # Build LIGHT Skeleton Schema (Table names only) to save tokens
        # Manager only needs to know which tables exist, not every column yet.
        skeleton_schema = f"Available Tables: {selected_tables}\nRelationships: {fk_relationships}"
        
        prompt_template = get_decomposer_prompt()
        # Manager uses 8B Flash for high-speed planning.
        chain = prompt_template | llm.with_structured_output(DecomposerOutput)
        res = robust_invoke(chain, {
            "skeleton_schema": skeleton_schema,
            "question": user_question
        }, DecomposerOutput)
        res.node_name = "decomposer"
        
        # --- DETERMINISTIC JOIN KEY INJECTION ---
        # Ensure every column mentioned in the JoinPlan steps is in the required_columns of the relevant sub-tasks.
        task_map = {t.task_id: t for t in res.sub_tasks}
        for step in res.join_plan.steps:
            for task_id in [step.left, step.right]:
                if task_id in task_map:
                    if step.on not in task_map[task_id].required_columns:
                        logger.info(f"Injecting missing join key '{step.on}' into {task_id}")
                        task_map[task_id].required_columns.append(step.on)

        logger.info(f"Decomposition complete: {len(res.sub_tasks)} tasks planned. Complexity: {res.complexity_score}")
        
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())
        
        # Convert Pydantic models to Dicts for state storage
        sub_tasks_dict = [t.model_dump() for t in res.sub_tasks]
        join_plan_dict = res.join_plan.model_dump()
        
        return {
            "sub_tasks": sub_tasks_dict,
            "join_plan": join_plan_dict,
            "sql_snippets": {}, # Initialize snippet storage
            "current_task_index": 0,
            "agent_logs": logs
        }
    finally:
        log_context.reset(token)

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
        # Use json_mode for structured output
        chain = prompt_template | llm.with_structured_output(ChatbotResponse)
        
        logger.info(f"Chatbot Node | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")
        
        res = robust_invoke(chain, {
            "lessons": lessons_text,
            "messages": current_chat_history
        }, ChatbotResponse)

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

### CLASSIFICATION RULES (Output JSON keys: "intent", "thought_process"):
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

### CLASSIFICATION RULES (Output JSON keys: "intent", "thought_process"):
- SQL: The request is a clear, actionable question about querying or analyzing data from the domain.
- CLARIFY: The request is related to the domain but is too vague, ambiguous, or underspecified to generate SQL for (e.g., "Tell me about films" without criteria).
- DENY: The request is NOT about the domain, asks to MODIFY data, or is general chitchat.

User Message: "{last_msg}"
"""
        # Use structured output for determinism
        chain = llm.with_structured_output(GuardianOutput)
        res = robust_invoke(chain, [SystemMessage(content=decision_prompt)], GuardianOutput)
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
        # Get last 3 human messages for context (handles "try again" or follow-ups)
        human_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-3:]
        context_msg = " | ".join(human_msgs)
        last_msg = human_msgs[-1] if human_msgs else ""
        
        logger.info(f"Node: classifier_node | Classifying message: {last_msg[:30]} | Context: {context_msg[:50]}...")
        
        prompt = f"""You are a SQL Query Planner. 
Analyze the user interaction and determine if the current request requires joining multiple tables (COMPLEX) or if it is a single-table query (SIMPLE).

### CONTEXT (Last messages):
{context_msg}

### CURRENT QUESTION TO CLASSIFY:
"{last_msg}"

### EXAMPLES:
- "What are the first 10 films?" -> SIMPLE
- "Canada Action films" -> COMPLEX
- "try again" (where previous was Canada films) -> COMPLEX
- "count them" (where previous was customers) -> SIMPLE

### GUIDELINES (Output JSON keys: "is_complex", "thought_process"):
- If the current message is a follow-up like "try again", "run it", or "go ahead", use the PREVIOUS context to decide.
- If any geographic (Country/City) or category filters were mentioned recently, it is COMPLEX.
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
    prompt = f"The user asked: '{state['messages'][-1].content}'. This is too vague for a SQL query. Ask a concise follow-up question to clarify what they want to see from the DVD rental database. Output MUST be valid JSON with key: 'clarification_question'."
    res = robust_invoke(chain, prompt, ClarificationOutput)
    
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
        entity_prompt = f"""Identify the core database entities and filters mentioned in this question. Output MUST be valid JSON with keys: "anchors", "thought_process".
Question: "{last_msg}"
Example Entities: 'Canada', 'Action', 'most rentals', 'spent'.
"""
        entity_chain = llm.with_structured_output(AnchorSelection)
        entity_res = entity_chain.invoke([SystemMessage(content=entity_prompt)])
        entities = ", ".join(entity_res.anchors)

        # Pass 2: Hard Physical Table Mapping
        mapping_prompt = f"""You are a Database Architect. Map the following entities to the specific PHYSICAL tables needed to query them. Output MUST be valid JSON with keys: "anchors", "thought_process".
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
        
        pruning_prompt = f"""You are a Data Architect. Prune the schema below to ONLY include the columns needed for this question. Output MUST be valid JSON with keys: "selected_tables", "selected_columns", "fk_relationships", "fk_path_identified", "thought_process".
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
        res = robust_invoke(chain, [SystemMessage(content=pruning_prompt)], SchemaSelectorOutput)
        res.node_name = "column_pruner"
        
        # Convert List[ColumnSelection] to Dict[str, List[str]] for state compatibility
        pruned_cols = {item.table_name: item.columns for item in res.selected_columns}
        # Convert List[FKRelationship] to List[Dict] for state compatibility
        pruned_fks = [rel.model_dump() for rel in res.fk_relationships]
        
        # Deterministic Guard: Ensure all selected tables exist and Join Keys are preserved
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
            "fk_relationships": pruned_fks,
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
        
        full_schema = sql_engine.get_schema_object()
        if selected_tables:
            filtered_schema = {t: full_schema.get(t, []) for t in selected_tables}
            schema_str = str(filtered_schema)
        else:
            schema_str = str(full_schema)
        
        prompt_template = get_sql_healing_prompt()
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

def worker_node(state: State, config: RunnableConfig):
    """
    The ReliableWorker Node: Solves an atomic sub-task.
    """
    configurable = config.get("configurable", {})
    task = state.get("current_task") 
    if not task:
        idx = state.get("current_task_index", 0)
        if idx < len(state.get("sub_tasks", [])):
            task = state["sub_tasks"][idx]
    
    if not task:
        logger.error(f"Worker Node received empty task.")
        return {"sql_error": "No task assigned to worker"}

    task_id = task.get("task_id") or task.get("id")
    description = task.get("description") or "No description"
    
    logger.info(f"Node: worker_node | Task: {task_id} ({description})")
    
    try:
        selected_tables = task.get("tables", [])
        required_columns = task.get("required_columns", [])
        partial_schema = sql_engine.get_schema(selected_tables)
        
        prompt = f"""You are a Reliable SQL Worker. Solve the following ATOMIC sub-task for the Pagila database.

TASK: "{description}"
REQUIRED JOIN KEYS: {required_columns} (These MUST be in your SELECT list)
SCHEMA:
{partial_schema}

RULES:
- Return ONLY valid PostgreSQL.
- No semicolons.
- Use explicit aliases (e.g., 't.' for table name).
- Ensure ALL {required_columns} are in the SELECT list so they can be joined later.
- STRICT SCHEMA ADHERENCE: Do NOT use columns not listed in the SCHEMA for a given table.
"""
        chain = llm.with_structured_output(WorkerOutput)
        res = chain.invoke([SystemMessage(content=prompt)])
        sql = res.sql.strip().replace("```sql", "").replace("```", "")
        
        try:
            sqlglot.parse_one(sql, read="postgres")
            return {
                "sql_snippets": {task_id: sql},
                "agent_logs": [res.model_dump()]
            }
        except Exception as parse_err:
            return {
                "sql_error": f"Task {task_id} syntax error: {str(parse_err)}",
                "current_sql": sql,
                "agent_logs": [res.model_dump()]
            }
    except Exception as e:
        return {"sql_error": f"Worker error: {str(e)}"}

def assembler_node(state: State):
    """
    The Assembler Node: Deterministically merges all CTE snippets using SQLTranspiler.
    """
    logger.info("Node: assembler_node")
    snippets = state.get("sql_snippets", {})
    join_plan = state.get("join_plan", {})
    
    try:
        final_sql = SQLTranspiler.merge_snippets(snippets, join_plan)
        logger.info("SQL Assembly Successful.")
        
        logs = state.get("agent_logs", [])
        logs.append({
            "node_name": "assembler",
            "thought_process": f"Stitched {len(snippets)} snippets using Join Plan.",
            "final_sql": final_sql
        })
        
        return {"current_sql": final_sql, "agent_logs": logs}
    except Exception as e:
        logger.error(f"SQL Assembly Failed: {e}")
        return {"sql_error": f"Assembly Error: {str(e)}"}

def format_sql_response_node(state: State, config: RunnableConfig):
    """
    Ultra-low latency renderer using Flash (8B) for summaries and Python for tables.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        raw_results = state.get("sql_results", [])
        is_aggregated = state.get("is_aggregated", False)

        sample_data = raw_results[:5] if len(raw_results) > 5 else raw_results
        
        prompt_template = get_sql_response_format_prompt()
        chain = prompt_template | llm.with_structured_output(SQLResponse)
        
        try:
            response = chain.invoke({
                "question": user_question,
                "query": state["current_sql"],
                "data": str(sample_data)
            })
            
            output_parts = []
            if response.summary:
                output_parts.append(response.summary)
            elif is_aggregated and raw_results:
                val = list(raw_results[0].values())[0]
                output_parts.append(f"The result is {val}.")

            if not is_aggregated and raw_results:
                table_md = generate_markdown_table(raw_results)
                output_parts.append(table_md)
                
            if state.get("current_sql"):
                sql_block = f"**Executed SQL:**\n```sql\n{state['current_sql'].strip()}\n```"
                output_parts.append(sql_block)
                
            final_content = "\n\n".join(output_parts)
            return {"messages": [AIMessage(content=final_content or "No results found.")]}
            
        except Exception as e:
            return {"messages": [AIMessage(content="I encountered an error while formatting the data.")]}
    finally:
        log_context.reset(token)

def _background_distill_lesson(state_data: dict, store, user_id: str):
    """Background task for lesson distillation."""
    try:
        learning_prompt = f"""You are a Senior Staff Engineer mentoring junior agents.
Create a "Golden standard Lesson" from this SQL mistake. Output MUST be valid JSON with keys: "is_global", "tags", "title", "body", "ending_note", "thought_process".
FAIL: {state_data['current_sql']}
ERROR: {state_data['sql_error']}
FIX: {state_data['fixed_sql']}
"""
        distiller = llm.with_structured_output(LessonDistillationOutput)
        lesson = distiller.invoke([SystemMessage(content=learning_prompt)])
        
        record_lesson(
            lesson.title, 
            state_data['sql_error'], 
            lesson.body.instruction, 
            lesson.body.reasoning, 
            store, 
            is_global=lesson.is_global,
            tags=state_data.get('selected_tables')
        )
    except Exception as e:
        logger.error(f"Background lesson distillation failed: {e}")
