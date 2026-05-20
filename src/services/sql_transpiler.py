import sqlglot
from sqlglot import exp
from src.workflow.schema import QueryBlueprint, SQLSelection, SQLFilter
from src.utils.logger import logger
from typing import Any

class SQLTranspiler:
    """
    Deterministic transpiler that converts a QueryBlueprint object 
    into dialect-specific SQL using SQLGlot.
    """
    
    @staticmethod
    def _parse_column(col_str: str) -> exp.Column:
        """Helper to parse 'table.column' into a SQLGlot Column object."""
        if col_str == "*":
            return exp.Star()
        if "." in col_str:
            parts = col_str.split(".")
            return exp.column(parts[1], table=parts[0])
        return exp.column(col_str)

    @staticmethod
    def to_sql(blueprint: QueryBlueprint, dialect: str = "postgres") -> str:
        """
        Converts the logical blueprint tree into a SQL string.
        """
        # --- FALLBACK: Direct SQL ---
        if hasattr(blueprint, "sql") and blueprint.sql:
            logger.info("Using direct SQL fallback from blueprint.")
            # Remove any markdown wrappers if model hallucinated them into the field
            sql = blueprint.sql.strip().replace("```sql", "").replace("```", "").rstrip(";")
            return sql

        logger.info(f"Transpiling QueryBlueprint for task {blueprint.task_id} to {dialect}")
        
        try:
            # 1. Start with SELECT
            selections = []
            for s in blueprint.select:
                col_expr = SQLTranspiler._parse_column(s.column)
                
                # Handle aggregation
                if s.aggregation:
                    agg_class = getattr(exp, s.aggregation.capitalize(), None)
                    if not agg_class:
                        agg_class = getattr(exp, s.aggregation.upper(), None)
                    
                    if agg_class:
                        expr = agg_class(this=col_expr)
                    else:
                        expr = exp.Identifier(this=f"{s.aggregation}({s.column})", quoted=False)
                else:
                    expr = col_expr
                
                # Handle alias
                if s.alias:
                    expr = exp.Alias(this=expr, alias=exp.Identifier(this=s.alias, quoted=False))
                
                selections.append(expr)
            
            query = exp.select(*selections)
            
            # 2. FROM (and basic joins)
            if blueprint.tables:
                query = query.from_(blueprint.tables[0])
                for table in blueprint.tables[1:]:
                    # For simple objectification, we use USING if the schema selector 
                    # implies a standard join, or just a generic JOIN.
                    # We'll use comma-less JOIN syntax for better readability.
                    query = query.join(table, join_type="INNER")
            
            # 3. WHERE
            for f in (blueprint.filters or []):
                col_expr = SQLTranspiler._parse_column(f.field)
                
                op_map = {
                    "=": exp.EQ,
                    "!=": exp.NEQ,
                    ">": exp.GT,
                    ">=": exp.GTE,
                    "<": exp.LT,
                    "<=": exp.LTE,
                    "LIKE": exp.Like,
                    "ILIKE": exp.ILike
                }
                
                if f.operator == "IS NULL":
                    condition = exp.Is(this=col_expr, expression=exp.Null())
                elif f.operator == "IN":
                    values = [exp.Literal.string(v) if isinstance(v, str) else exp.Literal.number(v) for v in f.value]
                    condition = exp.In(this=col_expr, expressions=values)
                else:
                    op_class = op_map.get(f.operator)
                    if op_class:
                        val_expr = exp.Literal.string(f.value) if isinstance(f.value, str) else exp.Literal.number(f.value)
                        condition = op_class(this=col_expr, expression=val_expr)
                    else:
                        condition = exp.Identifier(this=f"{f.field} {f.operator} {f.value}", quoted=False)
                
                query = query.where(condition)
            
            # 4. GROUP BY
            if blueprint.group_by:
                query = query.group_by(*[exp.column(c) for c in blueprint.group_by])
            
            # 5. ORDER BY
            if blueprint.order_by:
                order_exprs = []
                for order_dict in blueprint.order_by:
                    for col, direction in order_dict.items():
                        order_exprs.append(exp.Ordered(this=exp.column(col), desc=(direction == "DESC")))
                query = query.order_by(*order_exprs)
            
            # 6. LIMIT
            if blueprint.limit:
                query = query.limit(blueprint.limit)
            
            # Final generation
            sql_string = query.sql(dialect, pretty=True)
            logger.debug(f"Transpiled SQL: {sql_string}")
            return sql_string
            
        except Exception as e:
            logger.error(f"SQL Transpilation Failed: {e}", exc_info=True)
            raise ValueError(f"Could not transpile QueryBlueprint: {e}")

    @staticmethod
    def merge_snippets(snippets: dict[str, str], join_plan: dict[str, Any]) -> str:
        """
        Stitches multiple SQL snippets (CTEs) into a single valid PostgreSQL query.
        """
        logger.info(f"Assembling SQL snippets using Join Plan: {join_plan.get('base_task')}")
        
        # 1. Initialize the base query
        base_task_id = join_plan.get("base_task")
        if not base_task_id or base_task_id not in snippets:
            raise ValueError(f"Base task ID '{base_task_id}' not found in provided snippets.")
            
        final_select = join_plan.get("final_select", "*")
        
        # SQLGlot select() needs individual expressions. If it's a string with commas, we split it.
        if isinstance(final_select, str) and "," in final_select:
            select_exprs = [s.strip() for s in final_select.split(",")]
        else:
            select_exprs = [final_select] if isinstance(final_select, str) else final_select

        final_query = sqlglot.select(*select_exprs).from_(base_task_id)
        
        # 2. Add Join Steps
        for step in join_plan.get("steps", []):
            left = step.get("left")
            right = step.get("right")
            on_col = step.get("on")
            jtype = step.get("join_type", "inner")
            
            # Construct join condition: left.col = right.col
            join_cond = f"{left}.{on_col} = {right}.{on_col}"
            
            if jtype == "left":
                final_query = final_query.join(right, on=join_cond, join_type="LEFT")
            elif jtype == "cross":
                final_query = final_query.join(right, join_type="CROSS")
            else:
                final_query = final_query.join(right, on=join_cond)

        # 3. Inject Snippets as CTEs (Deterministic isolation)
        for task_id, raw_sql in snippets.items():
            # Clean up the raw SQL (remove trailing semicolons if any)
            clean_sql = raw_sql.strip().rstrip(";")
            
            # Use SQLGlot to wrap the snippet as a CTE
            final_query = final_query.with_(task_id, as_=sqlglot.parse_one(clean_sql, read="postgres"))
            
        return final_query.sql(dialect="postgres", pretty=True)
