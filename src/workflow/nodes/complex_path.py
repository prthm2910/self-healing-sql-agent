# ### --- [IMPORTS] --- ###

import sqlglot
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.utils.logger import logger
from src.services.sql_engine import sql_engine
from src.services.sql_assembler import sql_assembler
from src.prompts.sql_agent import get_decomposer_prompt, get_worker_prompt
from src.workflow.schema.complex_path import DecomposerOutput, WorkerOutput
from src.workflow.nodes.base import BaseNode, llm


# ### --- [DECOMPOSER NODE] --- ###

class DecomposerNode(BaseNode):
    """
    The Manager Node: Decomposes complex queries into atomic sub-tasks and a Join Plan.
    
    Utilizes 'Skeleton Knowledge' (Tables + FK Paths) from the Anchor Selector to 
    plan independent query 'islands' (sub-tasks). It enforces deterministic join key
    injections, ensuring that every island projects the exact column mappings required
    by downstream stitching.
    """
    name: str = "decomposer_node (Manager)"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Decompose a user request into discrete tasks and a relational join schema.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State updates with planned sub_tasks and join_plan.
        """
        # 1. Question Extraction: Locate the latest natural language message from the user.
        user_question: str = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), 
            ""
        )
        
        # 2. Extract Active Anchors and Bridges: Load selected tables and FK relationships from discovery.
        selected_tables: List[str] = state.get("selected_tables", [])
        fk_relationships: List[Dict[str, Any]] = state.get("fk_relationships", [])
        
        # 3. Build Token-Light Skeleton: We pass only the schema table names and active FK linkages
        # to save context window tokens, since complete DDLs are deferred to individual worker nodes.
        skeleton_schema: str = f"Available Tables: {selected_tables}\nRelationships: {fk_relationships}"
        
        # 4. Invoke Decomposer Planning: Compile the user request and schema outline.
        prompt_template = get_decomposer_prompt()
        chain = prompt_template | llm.with_structured_output(DecomposerOutput)
        res: DecomposerOutput = self.robust_invoke(chain, {
            "skeleton_schema": skeleton_schema,
            "question": user_question
        }, DecomposerOutput)
        res.node_name = "decomposer"
        
        # --- DETERMINISTIC JOIN KEY INJECTION ---
        # Mental Model:
        # If the LLM partitions a complex query into "task_1" (actor + film_actor) and "task_2" (film),
        # but fails to select the common join key (e.g. `film_id`) in one of the tasks' projections,
        # downstream CTE stitching is impossible and fails with compilation errors.
        #
        # Solution:
        # We loop through the generated join plan. For every step linking two tasks, we verify that the
        # required join column (`step.on`) is explicitly present in both tasks' required_columns list.
        # If missing, we inject it deterministically.
        task_map = {t.task_id: t for t in res.sub_tasks}
        for step in res.join_plan.steps:
            for task_id in [step.left, step.right]:
                if task_id in task_map:
                    if step.on not in task_map[task_id].required_columns:
                        logger.info(f"Injecting missing join key '{step.on}' into {task_id}")
                        task_map[task_id].required_columns.append(step.on)

        logger.info(f"Decomposition complete: {len(res.sub_tasks)} tasks planned. Complexity: {res.complexity_score}")
        
        # 5. Observability logging
        logs: List[Dict[str, Any]] = state.get("agent_logs", [])
        logs.append(res.model_dump())
        
        # 6. Convert models to standard dict configurations compatible with graph state boundaries
        sub_tasks_dict: List[Dict[str, Any]] = [t.model_dump() for t in res.sub_tasks]
        join_plan_dict: Dict[str, Any] = res.join_plan.model_dump()
        
        return {
            "sub_tasks": sub_tasks_dict,
            "join_plan": join_plan_dict,
            "sql_snippets": {},  # Initialize snippet storage
            "current_task_index": 0,
            "agent_logs": logs
        }


# ### --- [WORKER NODE] --- ###

class WorkerNode(BaseNode):
    """
    The ReliableWorker Node: Solves an atomic sub-task.
    
    Acts as a specialized query engine node dedicated to generating optimal PostgreSQL
    snippets for a single database 'island'. Leverages pre-pruned table schemas and 
    specifically injected join columns to write high-performance subqueries in isolation.
    """
    name: str = "worker_node"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Solves an assigned sub-task and parses the generated SQL.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes with generated SQL snippet or error descriptions.
        """
        # 1. Fetch Assigned Task: Retrieve the current active sub-task from workflow state.
        task: Optional[Dict[str, Any]] = state.get("current_task") 
        if not task:
            idx: int = state.get("current_task_index", 0)
            if idx < len(state.get("sub_tasks", [])):
                task = state["sub_tasks"][idx]
        
        # 2. Zero-check Validation
        if not task:
            logger.error("Worker Node received empty task.")
            return {"sql_error": "No task assigned to worker"}

        task_id: str = task.get("task_id") or task.get("id") or "unknown_task"
        description: str = task.get("description") or "No description"
        
        logger.info(f"Node: worker_node | Task: {task_id} ({description})")
        
        try:
            # 3. Retrieve Table Metadata: Load exact, clean column schema lists for tables in this sub-task.
            selected_tables: List[str] = task.get("tables", [])
            required_columns: List[str] = task.get("required_columns", [])
            partial_schema: Dict[str, List[str]] = sql_engine.get_schema(selected_tables)
            
            # 4. Generate isolated SQL Subquery: Request LLM to build a standalone SELECT query.
            prompt_template = get_worker_prompt()
            prompt_val = prompt_template.invoke({
                "required_columns": required_columns,
                "schema": partial_schema,
                "task_description": description
            })
            chain = llm.with_structured_output(WorkerOutput)
            res: WorkerOutput = self.robust_invoke(chain, prompt_val.to_messages(), WorkerOutput)
            
            # Cleanse markdown fences if present
            sql: str = res.sql.strip().replace("```sql", "").replace("```", "")
            
            # 5. AST Syntax Check: Proactively validate generated subquery using SQLGlot parser.
            # This isolates errors locally before stitching, preventing malformed segments from entering the assembler.
            try:
                sqlglot.parse_one(sql, read="postgres")
                return {
                    "sql_snippets": {task_id: sql},
                    "agent_logs": [res.model_dump()]
                }
            except Exception as parse_err:
                # Syntax error detected: return segment error to trigger self-healing loops
                return {
                    "sql_error": f"Task {task_id} syntax error: {str(parse_err)}",
                    "current_sql": sql,
                    "agent_logs": [res.model_dump()]
                }
        except Exception as e:
            return {"sql_error": f"Worker error: {str(e)}"}


# ### --- [ASSEMBLER NODE] --- ###

class AssemblerNode(BaseNode):
    """
    The Assembler Node: Deterministically merges all CTE snippets using SQLAssembler.
    
    Synthesizes the complete final CTE query using SQLGlot AST stitching. It prevents
    namespace collisions by applying local table prefix names and constructs clean standard
    nested clauses in compliance with the join plan.
    """
    name: str = "assembler_node"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Merges generated subquery snippets using the global sql_assembler tool.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: Fully stitched CTE query or detailed compilation errors.
        """
        # 1. Load compiled snippets and relational join plan.
        snippets: Dict[str, str] = state.get("sql_snippets", {})
        join_plan: Dict[str, Any] = state.get("join_plan", {})
        
        try:
            # 2. Delegate to SQLAssembler: Executes AST-stitching compilation.
            # Merges isolated query snippets into a unified, valid PostgreSQL Common Table Expression (CTE).
            final_sql: str = sql_assembler.assemble(snippets, join_plan)
            logger.info("SQL Assembly Successful.")
            
            # 3. Observability Logging
            logs: List[Dict[str, Any]] = state.get("agent_logs", [])
            logs.append({
                "node_name": "assembler",
                "thought_process": f"Stitched {len(snippets)} snippets using Join Plan.",
                "final_sql": final_sql
            })
            
            return {"current_sql": final_sql, "agent_logs": logs}
        except Exception as e:
            logger.error(f"SQL Assembly Failed: {e}")
            return {"sql_error": f"Assembly Error: {str(e)}"}


# ### --- [NODE INSTANTIATION] --- ###

# Instantiate node callable objects
decomposer_node = DecomposerNode()
worker_node = WorkerNode()
assembler_node = AssemblerNode()


