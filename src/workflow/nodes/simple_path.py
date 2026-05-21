import threading
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.services.sql_engine import sql_engine
from src.services.lessons import get_relevant_lessons
from src.prompts.sql_agent import get_sql_generation_prompt, get_sql_healing_prompt
from src.workflow.schema.simple_path import SQLGenerationOutput, ExecuteSQLOutput
from src.workflow.nodes.base import BaseNode, llm, logger
from src.workflow.nodes.background_tasks import background_distill_lesson


class GenerateSQLNode(BaseNode):
    """Generates SQL query using surgically pruned schema and tiered lessons."""
    name = "generate_sql"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
        user_question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        
        selected_columns = state.get("selected_columns")
        selected_tables = state.get("selected_tables")
        
        if selected_columns is not None:
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
        
        res = self.robust_invoke(chain, {
            "schema": schema_str,
            "lessons": lessons_text,
            "history": state["messages"][:-1],
            "question": user_question
        }, SQLGenerationOutput)
        
        sql_query = res.sql.strip().replace("```sql", "").replace("```", "")
        return {"current_sql": sql_query, "retry_count": 0}


class ExecuteSQLNode(BaseNode):
    """
    Executes the current SQL query and captures results or errors.
    """
    name = "execute_sql"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        current_sql = state.get("current_sql")
        if not current_sql:
            error_msg = state.get("sql_error") or "No SQL query generated."
            logger.error(f"Node: execute_sql | Missing Query! Error: {error_msg}")
            return {"sql_error": error_msg}

        logger.info(f"Node: execute_sql | Query: {current_sql}")
        raw_result = sql_engine.execute_query(current_sql)
        
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


class HealSQLNode(BaseNode):
    """Heals SQL and offloads lesson distillation to a background task."""
    name = "heal_sql"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
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
        res = self.robust_invoke(chain, {
            "schema": schema_str,
            "failed_query": state["current_sql"],
            "error_message": state["sql_error"],
            "question": user_question
        }, SQLGenerationOutput)
        
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
                target=background_distill_lesson,
                args=(state_data, store, user_id),
                daemon=True
            ).start()
            logger.info("Lesson distillation offloaded to background thread.")

        return {"current_sql": fixed_sql, "retry_count": retry, "sql_error": None}


# Instantiate node callable objects
generate_sql_node = GenerateSQLNode()
execute_sql_node = ExecuteSQLNode()
heal_sql_node = HealSQLNode()
