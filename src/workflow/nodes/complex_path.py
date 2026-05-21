import sqlglot
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.utils.logger import logger
from src.services.sql_engine import sql_engine
from src.services.sql_assembler import sql_assembler
from src.prompts.sql_agent import get_decomposer_prompt, get_worker_prompt
from src.workflow.schema.complex_path import DecomposerOutput, WorkerOutput
from src.workflow.nodes.base import BaseNode, llm


class DecomposerNode(BaseNode):
    """
    The Manager Node: Decomposes complex queries into atomic sub-tasks and a Join Plan.
    Utilizes 'Skeleton Knowledge' (Tables + FK Paths) from the Anchor Selector.
    """
    name = "decomposer_node (Manager)"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        user_question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        
        selected_tables = state.get("selected_tables", [])
        fk_relationships = state.get("fk_relationships", [])
        
        # Build LIGHT Skeleton Schema (Table names only) to save tokens
        skeleton_schema = f"Available Tables: {selected_tables}\nRelationships: {fk_relationships}"
        
        prompt_template = get_decomposer_prompt()
        chain = prompt_template | llm.with_structured_output(DecomposerOutput)
        res = self.robust_invoke(chain, {
            "skeleton_schema": skeleton_schema,
            "question": user_question
        }, DecomposerOutput)
        res.node_name = "decomposer"
        
        # --- DETERMINISTIC JOIN KEY INJECTION ---
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


class WorkerNode(BaseNode):
    """
    The ReliableWorker Node: Solves an atomic sub-task.
    """
    name = "worker_node"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
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
            
            prompt_template = get_worker_prompt()
            prompt_val = prompt_template.invoke({
                "required_columns": required_columns,
                "schema": partial_schema,
                "task_description": description
            })
            chain = llm.with_structured_output(WorkerOutput)
            res = self.robust_invoke(chain, prompt_val.to_messages(), WorkerOutput)
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


class AssemblerNode(BaseNode):
    """
    The Assembler Node: Deterministically merges all CTE snippets using SQLTranspiler.
    """
    name = "assembler_node"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        snippets = state.get("sql_snippets", {})
        join_plan = state.get("join_plan", {})
        
        try:
            final_sql = sql_assembler.assemble(snippets, join_plan)
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


# Instantiate node callable objects
decomposer_node = DecomposerNode()
worker_node = WorkerNode()
assembler_node = AssemblerNode()
