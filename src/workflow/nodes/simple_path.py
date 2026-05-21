# ### --- [IMPORTS] --- ###

import threading
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.services.sql_engine import sql_engine
from src.services.lessons import get_relevant_lessons
from src.prompts.sql_agent import get_sql_generation_prompt, get_sql_healing_prompt
from src.workflow.schema.simple_path import SQLGenerationOutput, ExecuteSQLOutput
from src.workflow.nodes.base import BaseNode, llm, logger
from src.workflow.nodes.background_tasks import background_distill_lesson


# ### --- [GENERATE SQL NODE] --- ###

class GenerateSQLNode(BaseNode):
    """
    Generates SQL query using surgically pruned schema and tiered lessons.
    
    This node serves as the primary SQL query planner for non-complex paths.
    It combines selected table lists or surgically pruned column maps with tiered
    semantic memory lessons to construct valid, highly optimized PostgreSQL queries.
    """
    name: str = "generate_sql"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        store: Optional[Any] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes query generation utilizing schema filtration and lesson stores.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes with generated SQL query and retry counter.
        """
        # 1. Question Extraction: Get the latest user natural language input.
        user_question: str = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), 
            ""
        )
        
        # 2. Extract Pruning Schema: Determine which schema level to supply to the LLM to minimize context size.
        selected_columns: Optional[Dict[str, List[str]]] = state.get("selected_columns")
        selected_tables: Optional[List[str]] = state.get("selected_tables")
        
        schema_str: str = ""
        if selected_columns is not None:
            # Scenario A (Highest Optimization): Inject only the surgically pruned columns discovered in discovery phase.
            logger.info(f"Using SURGICALLY PRUNED schema ({len(selected_columns)} tables)")
            schema_str = str(selected_columns)
        elif selected_tables:
            # Scenario B (Table filtered): Inject the full column catalogs for only the selected table set.
            logger.info(f"Using table-filtered schema ({len(selected_tables)} tables)")
            full_schema = sql_engine.get_schema_object()
            schema_str = str({t: full_schema.get(t, []) for t in selected_tables})
        else:
            # Scenario C (Fallback): Inject the entire database column map (only used for extremely basic setups).
            logger.info("Using full schema fallback")
            schema_str = str(sql_engine.get_schema_object())
        
        # 3. Tiered Lesson Retrieval:
        # We query the pgvector semantic store to load global rules, table-specific constraints, and 
        # semantic lessons matching past failures, caching lessons-learned to prevent query regressions.
        lessons_text, applied_titles = get_relevant_lessons(
            user_question, 
            store, 
            selected_tables=selected_tables
        )
        
        # 4. Invoke SQL Planner: Render prompt parameters and execute structured output call.
        prompt_template = get_sql_generation_prompt()
        chain = prompt_template | llm.with_structured_output(SQLGenerationOutput)
        
        res: SQLGenerationOutput = self.robust_invoke(chain, {
            "schema": schema_str,
            "lessons": lessons_text,
            "history": state["messages"][:-1],
            "question": user_question
        }, SQLGenerationOutput)
        
        # Clean any raw markdown formatting from generated SQL
        sql_query: str = res.sql.strip().replace("```sql", "").replace("```", "")
        
        # 5. State update: Reset retry count to 0 for initial query executions.
        return {"current_sql": sql_query, "retry_count": 0}


# ### --- [EXECUTE SQL NODE] --- ###

class ExecuteSQLNode(BaseNode):
    """
    Executes the current SQL query and captures results or errors.
    
    Provides standard execution safety boundaries by running raw generated SQL 
    queries against the database engine, returning structured outputs parsed and
    validated through the ExecuteSQLOutput Pydantic model.
    """
    name: str = "execute_sql"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes query statement and normalizes database driver responses.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: Normalized result records or error logs.
        """
        # 1. Fetch Planned SQL: Retrieve the active query string.
        current_sql: Optional[str] = state.get("current_sql")
        if not current_sql:
            error_msg: str = state.get("sql_error") or "No SQL query generated."
            logger.error(f"Node: execute_sql | Missing Query! Error: {error_msg}")
            return {"sql_error": error_msg}

        logger.info(f"Node: execute_sql | Query: {current_sql}")
        
        # 2. Database Driver Execution: Lease a connection and run query string against SQLEngine client.
        raw_result = sql_engine.execute_query(current_sql)
        
        # 3. Normalization and Contract checking: Map driver dictionary values to Pydantic structure.
        result = ExecuteSQLOutput(**raw_result)

        # 4. Process Driver Results
        if result.status == "success":
            data: List[Dict[str, Any]] = result.data
            
            # Aggregate Check: Determine if the result query outputs a single aggregated cell (e.g. SELECT count(*))
            is_aggregated: bool = False
            if len(data) == 1:
                row = data[0]
                if len(row.keys()) == 1:
                    is_aggregated = True
            
            logger.info(f"Execution Success. Rows: {result.row_count} | Aggregated: {is_aggregated}")
            return {"sql_results": data, "is_aggregated": is_aggregated, "sql_error": None}
        else:
            # Query failed: record error details to trigger healing nodes downstream.
            logger.warning(f"Execution Failed. Error: {result.error_message}")
            return {"sql_error": result.error_message, "sql_results": [], "is_aggregated": False}


# ### --- [HEAL SQL NODE] --- ###

class HealSQLNode(BaseNode):
    """
    Heals SQL and offloads lesson distillation to a background task.
    
    Executes standard self-healing feedback loop on SQL generation failures. Re-evaluates
    errors, schema metadata, and failed query outputs to reconstruct syntax corrections 
    while asynchronously invoking the lesson distillation queue in a separate daemon thread
    to prevent blocking core query pipelines.
    """
    name: str = "heal_sql"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        store: Optional[Any] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Launches query correcting loops and triggers background distillation.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes updates with corrected query statements.
        """
        # 1. Update retry counter: Track how many healing loops have run for active query.
        retry: int = state.get("retry_count", 0) + 1
        logger.info(f"Node: heal_sql | Attempt: {retry}")
        
        # 2. Context retrieval
        user_question: str = next(
            m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        selected_tables: Optional[List[str]] = state.get("selected_tables")
        
        # 3. Resolve active schema string to guide correct healing choices
        full_schema: Dict[str, List[str]] = sql_engine.get_schema_object()
        schema_str: str = ""
        if selected_tables:
            filtered_schema = {t: full_schema.get(t, []) for t in selected_tables}
            schema_str = str(filtered_schema)
        else:
            schema_str = str(full_schema)
        
        # 4. Invoke Healer Prompt: Pass the failed query and raw database error to the LLM to get a corrected query.
        prompt_template = get_sql_healing_prompt()
        chain = prompt_template | llm.with_structured_output(SQLGenerationOutput)
        res: SQLGenerationOutput = self.robust_invoke(chain, {
            "schema": schema_str,
            "failed_query": state["current_sql"],
            "error_message": state["sql_error"],
            "question": user_question
        }, SQLGenerationOutput)
        
        fixed_sql: str = res.sql.strip().replace("```sql", "").replace("```", "")
        
        # --- OFFLOAD TO BACKGROUND ---
        # Mental Model:
        # Lesson distillation queries the LLM to analyze the delta between the failed and fixed queries.
        # This takes 1-3 seconds. To prevent the end-user from experiencing this latency during active response
        # rendering, we offload this work asynchronously to an isolated background daemon thread.
        if store:
            state_data: Dict[str, Any] = {
                "current_sql": state["current_sql"],
                "sql_error": state["sql_error"],
                "fixed_sql": fixed_sql,
                "selected_tables": selected_tables,
                "user_question": user_question
            }
            threading.Thread(
                target=background_distill_lesson,
                args=(state_data, store, user_id),
                daemon=True
            ).start()
            logger.info("Lesson distillation offloaded to background thread.")

        # 5. Return fixed SQL and clear previous errors
        return {"current_sql": fixed_sql, "retry_count": retry, "sql_error": None}


# ### --- [NODE INSTANTIATION] --- ###

# Instantiate node callable objects
generate_sql_node = GenerateSQLNode()
execute_sql_node = ExecuteSQLNode()
heal_sql_node = HealSQLNode()


