import os
import atexit
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv
from langsmith import traceable

from src.utils.logger import logger

load_dotenv()

class SQLEngine:
    def __init__(self, db_name: str = "pagila"):
        self.db_name = db_name
        self._pool = None
        base_url = os.getenv("DATABASE_URL")
        if not base_url:
            logger.error("DATABASE_URL not found in environment")
            raise ValueError("DATABASE_URL not found in environment")
        
        # Create a dedicated pool for the specific database (e.g. pagila)
        parsed = urlparse(base_url)
        self.connection_url = urlunparse(parsed._replace(path=f"/{db_name}"))
        
        # Dynamic FK Mapping
        self._fk_map: Optional[Dict[str, Dict[str, str]]] = None
        self._graph: Optional[Dict[str, set]] = None
        self._schema_cache: Optional[Dict[str, List[str]]] = None

    @property
    def fk_map(self) -> Dict[str, Dict[str, str]]:
        if self._fk_map is None:
            self._fk_map = self._build_dynamic_fk_map()
        return self._fk_map

    @property
    def graph(self) -> Dict[str, set]:
        if self._graph is None:
            self._graph = {}
            for table, fks in self.fk_map.items():
                if table not in self._graph: self._graph[table] = set()
                for col, target_info in fks.items():
                    f_table = target_info["table"]
                    self._graph[table].add(f_table)
                    if f_table not in self._graph: self._graph[f_table] = set()
                    self._graph[f_table].add(table)
        return self._graph

    @traceable(name="Build Dynamic FK Map", run_type="tool")
    def _build_dynamic_fk_map(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Fetches foreign key relationships dynamically from PostgreSQL."""
        logger.info("Building dynamic FK map from information_schema")
        query = """
        SELECT
            tc.table_name, 
            kcu.column_name, 
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM 
            information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public';
        """
        fk_map = {}
        try:
            pool = self._get_pool()
            with pool.connection() as conn:
                results = conn.execute(query).fetchall()
                for row in results:
                    table = row["table_name"]
                    column = row["column_name"]
                    foreign_table = row["foreign_table_name"]
                    foreign_column = row["foreign_column_name"]
                    if table not in fk_map:
                        fk_map[table] = {}
                    fk_map[table][column] = {
                        "table": foreign_table,
                        "column": foreign_column
                    }
                return fk_map
        except Exception as e:
            logger.error(f"Failed to build dynamic FK map: {str(e)}")
            return {}

    @traceable(name="Identify FK Bridge Tables", run_type="tool")
    def get_bridge_tables(self, anchors: List[str]) -> List[str]:
        """
        Finds bridge tables needed to connect anchor tables using BFS.
        """
        if len(anchors) < 2:
            return []
            
        bridges = set()
        start_node = anchors[0]
        for target in anchors[1:]:
            path = self._find_shortest_path(start_node, target)
            if path:
                bridges.update(path)
        
        return list(bridges - set(anchors))

    @traceable(name="BFS Pathfinding", run_type="tool")
    def _find_shortest_path(self, start, end):
        """BFS implementation for shortest path."""
        if start == end: return [start]
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            (node, path) = queue.pop(0)
            for next_node in self.graph.get(node, []):
                if next_node == end:
                    return path + [next_node]
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        return None

    @traceable(name="Get Relevant FKs", run_type="tool")
    def get_relevant_fks(self, table_names: List[str]) -> List[Dict[str, str]]:
        """
        Fetches foreign key relationships between the specified tables.
        
        Returns:
            List[Dict]: List of FK definitions: {'source': table, 'col': col, 'target_table': table, 'target_col': col}
        """
        all_fks = self.fk_map
        relevant_fks = []
        for table in table_names:
            if table in all_fks:
                for col, target_info in all_fks[table].items():
                    if target_info["table"] in table_names:
                        relevant_fks.append({
                            "source_table": table,
                            "source_column": col,
                            "target_table": target_info["table"],
                            "target_column": target_info["column"]
                        })
        return relevant_fks
        
    def _get_pool(self):
        """Lazy initialization of the SQL connection pool."""
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self.connection_url,
                max_size=2,
                min_size=1,
                timeout=60.0,
                check=ConnectionPool.check_connection,
                kwargs={"autocommit": True, "row_factory": dict_row}
            )
            logger.debug(f"SQLEngine pool initialized for: {self.db_name}")
        return self._pool

    @traceable(name="Execute SQL Query", run_type="tool")
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Executes a SQL query using the lazy-loaded pool."""
        logger.info(f"Executing SQL Query: {query}")
        try:
            pool = self._get_pool()
            with pool.connection() as conn:
                results = conn.execute(query).fetchall()
                logger.info(f"Query successful. Rows returned: {len(results)}")
                return {"status": "success", "data": results, "row_count": len(results)}
        except Exception as e:
            logger.error(f"SQL Execution Failed. Error: {str(e)}")
            return {"status": "error", "error_message": str(e), "query": query}

    def _get_enum_map(self) -> Dict[str, List[str]]:
        """Fetches all ENUM types and their allowed values."""
        query = """
        SELECT
            t.typname AS enum_name,
            e.enumlabel AS enum_value
        FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'public'
        ORDER BY t.typname, e.enumsortorder;
        """
        enum_map = {}
        try:
            pool = self._get_pool()
            with pool.connection() as conn:
                results = conn.execute(query).fetchall()
                for row in results:
                    name = row["enum_name"]
                    val = row["enum_value"]
                    if name not in enum_map:
                        enum_map[name] = []
                    enum_map[name].append(val)
                return enum_map
        except Exception as e:
            logger.error(f"Failed to fetch ENUM map: {str(e)}")
            return {}

    def get_schema(self, table_names: Optional[List[str]] = None) -> str:
        """Returns schema with ENUM values resolved and complex type hints."""
        pool = self._get_pool()
        enum_map = self._get_enum_map()
        
        query = "SELECT table_name, column_name, data_type, udt_name FROM information_schema.columns WHERE table_schema = 'public'"
        if table_names:
            tables_str = ",".join([f"'{t}'" for t in table_names])
            query += f" AND table_name IN ({tables_str})"
        query += " ORDER BY table_name, ordinal_position"
        
        try:
            with pool.connection() as conn:
                columns = conn.execute(query).fetchall()
                
                schema_parts = []
                current_table = ""
                for col in columns:
                    if col["table_name"] != current_table:
                        current_table = col["table_name"]
                        schema_parts.append(f"\nTable: {current_table}")
                    
                    dtype = col["data_type"]
                    udt = col["udt_name"]
                    col_name = col["column_name"]
                    
                    # 1. Resolve ENUMs
                    if dtype == "USER-DEFINED" and udt in enum_map:
                        vals = ", ".join([f"'{v}'" for t in [udt] for v in enum_map[t]])
                        dtype = f"ENUM: {vals}"
                    
                    # 2. Add hints for ARRAYs (Specific to Pagila special_features)
                    elif dtype == "ARRAY":
                        if col_name == "special_features":
                            dtype = "ARRAY (Examples: 'Trailers', 'Commentaries', 'Deleted Scenes', 'Behind the Scenes')"
                        else:
                            dtype = "ARRAY"
                            
                    # 3. Add hints for Full-Text Search
                    elif dtype == "tsvector":
                        dtype = "tsvector (Full-Text Search Index)"
                    
                    schema_parts.append(f" - {col_name} ({dtype})")
                
                return "\n".join(schema_parts)
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

    @traceable(name="Get Schema Object", run_type="tool")
    def get_schema_object(self, table_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Returns the database schema as a structured dictionary: {table_name: [col1, col2, ...]}
        """
        # Return cache if available and no specific tables requested
        if not table_names and self._schema_cache:
            return self._schema_cache

        pool = self._get_pool()
        query = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
        if table_names:
            tables_str = ",".join([f"'{t}'" for t in table_names])
            query += f" AND table_name IN ({tables_str})"
        
        schema_obj = {}
        try:
            with pool.connection() as conn:
                results = conn.execute(query).fetchall()
                for row in results:
                    table = row["table_name"]
                    column = row["column_name"]
                    if table not in schema_obj:
                        schema_obj[table] = []
                    schema_obj[table].append(column)
                
                # Only update cache if it's the full schema
                if not table_names:
                    self._schema_cache = schema_obj
                return schema_obj
        except Exception as e:
            logger.error(f"Error fetching schema object: {str(e)}")
            return {}

    def list_tables(self) -> List[str]:
        pool = self._get_pool()
        try:
            with pool.connection() as conn:
                tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'").fetchall()
                return [t["table_name"] for t in tables]
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []

    def close(self):
        """Gracefully shuts down the connection pool."""
        if self._pool is not None:
            logger.info(f"Closing SQLEngine pool for: {self.db_name}")
            self._pool.close()
            self._pool = None

# Singleton instance
sql_engine = SQLEngine()

# Register for graceful shutdown
atexit.register(sql_engine.close)
