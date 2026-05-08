import sqlglot
from sqlglot import exp, parse_one
from typing import List, Dict, Any, Optional
from src.utils.logger import logger

class SQLAssembler:
    """
    Deterministic SQL Assembly Engine using SQLGlot ASTs.
    Implements the 'Divide & Conquer' final stitching logic.
    """
    
    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def assemble(
        self, 
        islands: Dict[str, str], 
        join_plan: List[Dict[str, Any]],
        final_select: List[str]
    ) -> str:
        """
        Stitches multiple SQL islands into a single CTE-based query.
        Handles automatic key injection, GROUP BY safety, and unique aliasing.
        """
        logger.info(f"Assembling {len(islands)} islands with {len(join_plan)} joins.")

        # 1. Map Required Keys per Island from Join Plan (Hardened)
        required_keys = {island_id: set() for island_id in islands.keys()}
        for join in join_plan:
            try:
                # FIX: Specify dialect for consistent parsing
                on_expr = sqlglot.parse_one(join["on"], read=self.dialect)
                for col in on_expr.find_all(exp.Column):
                    table_alias = col.table
                    col_name = col.this.name
                    
                    if table_alias in required_keys:
                        # FIX: Strip island prefix before identifying physical column
                        # e.g., 'island_actor_id' -> 'actor_id'
                        if col_name.startswith(f"{table_alias}_"):
                            col_name = col_name[len(table_alias) + 1:]
                        required_keys[table_alias].add(col_name)
            except Exception as e:
                logger.warning(f"Could not parse join keys from '{join['on']}': {e}")

        # 2. Add Islands as CTEs with Injection & Aliasing
        ctes = {}
        for island_id, sql in islands.items():
            try:
                island_ast = parse_one(sql, read=self.dialect)
                
                # FIX: Support Union/Except (Query types) for aliasing
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
                            # FIX: Prevent PostgreSQL aggregate errors
                            if island_ast.args.get("group"):
                                island_ast.group_by(key, copy=False)

                # B. Apply Unique Aliasing ({island_id}_{column_name})
                # Handle Selects and Unions/Set operations
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

        # 3. Build Final SELECT with Joins
        first_island_id = join_plan[0]["source"] if join_plan else list(islands.keys())[0]
        root = sqlglot.select(*final_select).from_(f"{first_island_id} AS {first_island_id}")

        for join in join_plan:
            root = root.join(
                f"{join['target']} AS {join['target']}", 
                on=join["on"], 
                join_type=join.get("join_type", "INNER").upper()
            )

        # 4. Attach CTEs
        for island_id, ast in ctes.items():
            root = root.with_(island_id, as_=ast)

        return root.sql(dialect=self.dialect, pretty=True)

# Singleton instance
sql_assembler = SQLAssembler(dialect="postgres")
