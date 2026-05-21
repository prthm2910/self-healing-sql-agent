from typing import List, Dict, Any, Optional

import sqlglot
from sqlglot import exp, parse_one

from src.utils.logger import logger


class SQLAssembler:
    """
    Deterministic SQL Assembly Engine using SQLGlot ASTs.
    
    Implements the 'Divide & Conquer' final stitching logic by treating 
    independent SQL 'islands' as Common Table Expressions (CTEs) and 
    joining them according to a structured plan.

    Examples:
        >>> islands = {
        ...     "actor": "SELECT first_name FROM actor",
        ...     "films": "SELECT actor_id, COUNT(*) as cnt FROM film_actor GROUP BY actor_id"
        ... }
        >>> plan = [{
        ...     "source": "actor", 
        ...     "target": "films", 
        ...     "on": "actor.actor_actor_id = films.films_actor_id"
        ... }]
        >>> assembler = SQLAssembler()
        >>> sql = assembler.assemble(islands, plan, ["actor_first_name", "films_cnt"])
        >>> print(sql)
        WITH actor AS (SELECT first_name, actor_id AS actor_actor_id FROM actor),
             films AS (SELECT actor_id, COUNT(*) AS cnt, actor_id AS films_actor_id ...)
        SELECT actor_first_name, films_cnt FROM actor JOIN films ...
    """
    
    def __init__(self, dialect: str = "postgres"):
        """
        Initialize the assembler with a specific SQL dialect.

        Args:
            dialect (str): The SQLGlot dialect to use (e.g., "postgres", "mysql").
        """
        self.dialect = dialect

    def assemble(
        self, 
        islands: Dict[str, str], 
        join_plan: Any,
        final_select: Optional[List[str]] = None
    ) -> str:
        """
        Stitches multiple SQL islands into a single CTE-based query.

        Handles automatic key injection to support joins, ensures GROUP BY 
        compatibility for injected columns, and enforces unique aliasing 
        to prevent namespace collisions.

        Args:
            islands (Dict[str, str]): Map of island names to their partial SQL.
            join_plan (Union[Dict[str, Any], List[Dict[str, Any]]]): Structured join instructions
                (JoinPlan dict) or a list of join definitions (legacy).
            final_select (Optional[List[str]]): List of column names to project in the 
                final query (legacy).

        Returns:
            str: The fully assembled, pretty-printed SQL string.

        Raises:
            ValueError: If an island contains invalid SQL or if injection fails.
        """
        # Compatibility wrapper for the old list-based join plan interface
        if isinstance(join_plan, list):
            base_task = join_plan[0]["source"] if join_plan else list(islands.keys())[0]
            steps = []
            for j in join_plan:
                steps.append({
                    "left": j["source"],
                    "right": j["target"],
                    "on": j["on"],
                    "join_type": j.get("join_type", "INNER")
                })
            plan_dict = {
                "base_task": base_task,
                "steps": steps,
                "final_select": final_select if final_select else ["*"]
            }
        else:
            plan_dict = join_plan

        logger.info(f"Assembling {len(islands)} islands with JoinPlan: {plan_dict.get('base_task')}")

        # ### --- [STEP 1: KEY DISCOVERY] --- ###

        # [Elaborative Breakdown]
        # This phase traverses the join plan steps to identify which columns (keys) 
        # are required for the islands to connect. It handles the 'Naming Mismatch'
        # by stripping the island-prefixed aliases used in the join plan to 
        # find the actual physical column names for injection.
        
        required_keys = {island_id: set() for island_id in islands.keys()}
        for step in plan_dict.get("steps", []):
            try:
                # Specify dialect for consistent parsing
                on_expr = sqlglot.parse_one(step["on"], read=self.dialect)
                for col in on_expr.find_all(exp.Column):
                    table_alias = col.table
                    col_name = col.this.name
                    
                    if table_alias in required_keys:
                        # Strip island prefix before identifying physical column
                        # e.g., 'island_actor_id' -> 'actor_id'
                        if col_name.startswith(f"{table_alias}_"):
                            col_name = col_name[len(table_alias) + 1:]
                        required_keys[table_alias].add(col_name)
            except Exception as e:
                logger.warning(f"Could not parse join keys from '{step.get('on')}': {e}")

        # ### --- [STEP 2: AST MANIPULATION] --- ###

        # [Elaborative Breakdown]
        # We iterate through each island, parsing it into a SQLGlot AST.
        # 1. We inject missing join keys. If the island is an aggregate query,
        #    we also inject the keys into the GROUP BY clause to maintain syntax correctness.
        # 2. We recursively apply unique aliasing ({island}_{column}) to all 
        #    projections, including those within nested SELECTs or Set operations (UNION).
        
        ctes = {}
        for island_id, sql in islands.items():
            try:
                island_ast = parse_one(sql, read=self.dialect)
                
                # Support Union/Except (Query types) for aliasing
                if not isinstance(island_ast, (exp.Select, exp.Query)):
                    ctes[island_id] = island_ast
                    continue
 
                # A. Inject Missing Keys (with GROUP BY safety)
                if isinstance(island_ast, exp.Select):
                    existing_cols = {c.alias_or_name for c in island_ast.expressions}
                    for key in required_keys.get(island_id, []):
                        if key not in existing_cols:
                            logger.debug(f"Surgically injecting key '{key}' into island '{island_id}'")
                            island_ast.select(key, copy=False)
                            # Prevent PostgreSQL aggregate errors by adding to GROUP BY if it exists
                            if island_ast.args.get("group"):
                                island_ast.group_by(key, copy=False)

                # B. Apply Unique Aliasing ({island_id}_{column_name})
                # Handle Selects and Unions/Set operations by finding all SELECT expressions
                selects = island_ast.find_all(exp.Select)
                for select in selects:
                    new_projections = []
                    for projection in select.expressions:
                        alias = projection.alias_or_name
                        unique_alias = f"{island_id}_{alias}"
                        new_projections.append(projection.as_(unique_alias))
                    select.set("expressions", new_projections)
                
                ctes[island_id] = island_ast
            except Exception as e:
                logger.error(f"Failed to process island {island_id}: {e}")
                raise ValueError(f"Invalid SQL in island {island_id}: {e}")

        # ### --- [STEP 3: FINAL STITCHING] --- ###

        try:
            # Initialize root query with the base_task
            base_task_id = plan_dict.get("base_task")
            if not base_task_id:
                base_task_id = list(islands.keys())[0]

            raw_select = plan_dict.get("final_select", "*")
            if isinstance(raw_select, str) and "," in raw_select:
                select_exprs = [s.strip() for s in raw_select.split(",")]
            elif isinstance(raw_select, str):
                select_exprs = [raw_select]
            else:
                select_exprs = raw_select

            root = sqlglot.select(*select_exprs).from_(f"{base_task_id} AS {base_task_id}")

            # Append joins from the plan steps
            for step in plan_dict.get("steps", []):
                right = step["right"]
                root = root.join(
                    f"{right} AS {right}", 
                    on=step["on"], 
                    join_type=step.get("join_type", "INNER").upper()
                )

            # Add Final Clauses (Where, Order By, Limit)
            if plan_dict.get("where"):
                root = root.where(plan_dict["where"])
                
            if plan_dict.get("order_by"):
                root = root.order_by(plan_dict["order_by"])
                
            if plan_dict.get("limit"):
                root = root.limit(plan_dict["limit"])

            # Attach all islands as CTEs
            for island_id, ast in ctes.items():
                root = root.with_(island_id, as_=ast)

            return root.sql(dialect=self.dialect, pretty=True)
        except Exception as e:
            logger.error(f"Failed to stitch query: {e}")
            raise ValueError(f"Stitching failed: {e}")

# ### --- [SINGLETON] --- ###

# Globally available assembler instance
sql_assembler = SQLAssembler(dialect="postgres")
