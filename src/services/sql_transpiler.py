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
        if "." in col_str:
            parts = col_str.split(".")
            return exp.column(parts[1], table=parts[0])
        return exp.column(col_str)

    @staticmethod
    def to_sql(blueprint: QueryBlueprint, dialect: str = "postgres") -> str:
        """
        Converts the logical blueprint tree into a SQL string.
        """
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
            for f in blueprint.filters:
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
