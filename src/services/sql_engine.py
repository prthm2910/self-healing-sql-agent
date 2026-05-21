# ### --- [IMPORTS & CONFIGURATION] --- ###
import os
import atexit
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import psycopg
import sqlglot
from sqlglot import exp, parse_one
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langsmith import traceable

from src.utils.logger import logger

load_dotenv()


# ### --- [SQL ENGINE CLIENT CLASS] --- ###

class SQLEngine:
    """An enterprise-grade, metadata-aware SQL compilation and execution service.

    Wraps a lazy-loaded ConnectionPool and dynamically queries PostgreSQL metadata 
    to enable schema reflection, custom ENUM resolution, and relation pathfinding.
    """

    def __init__(self, db_name: str = "pagila") -> None:
        """Initializes the engine, mapping target connection parameters from environment.

        Args:
            db_name: The name of the target database schema (defaults to 'pagila').

        Raises:
            ValueError: If DATABASE_URL is not found in the environment.
        """
        self.db_name: str = db_name
        self._pool: Optional[ConnectionPool] = None
        base_url: Optional[str] = os.getenv("DATABASE_URL")
        if not base_url:
            logger.error("DATABASE_URL not found in environment")
            raise ValueError("DATABASE_URL not found in environment")
        
        # Parse and surgical substitution to direct connections to the target database instance
        parsed = urlparse(base_url)
        self.connection_url: str = urlunparse(parsed._replace(path=f"/{db_name}"))
        
        # Dynamic cache maps used to build relationship bridges and schema skeletons
        self._fk_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._graph: Optional[Dict[str, Set[str]]] = None
        self._schema_cache: Optional[Dict[str, List[str]]] = None

    @property
    def fk_map(self) -> Dict[str, Dict[str, Any]]:
        """A lazy-loaded descriptor containing all active foreign keys in public schema.

        Returns:
            Dict[str, Dict[str, Any]]: Nested map: {table_name: {column_name: {table, column}}}
        """
        if self._fk_map is None:
            self._fk_map = self._build_dynamic_fk_map()
        return self._fk_map

    @property
    def graph(self) -> Dict[str, Set[str]]:
        """A lazy-loaded, undirected graph representation of the database table relationships.

        Returns:
            Dict[str, Set[str]]: Schema network: {table_name: {connected_table_names}}
        """
        if self._graph is None:
            self._graph = {}
            for table, fks in self.fk_map.items():
                if table not in self._graph: 
                    self._graph[table] = set()
                for col, target_info in fks.items():
                    f_table: str = target_info["table"]
                    self._graph[table].add(f_table)
                    if f_table not in self._graph: 
                        self._graph[f_table] = set()
                    self._graph[f_table].add(table)
        return self._graph


    # ### --- [DYNAMIC RELATIONSHIP MAPPING] --- ###

    # [Elaborative Breakdown]
    # Dynamic Schema Mapping:
    # Instead of hardcoding relationships or stuffing static SQL strings in configuration,
    # _build_dynamic_fk_map queries the PostgreSQL system catalog (`information_schema`)
    # on startup. It extracts active key constraint tables, column bridges, and targets,
    # building a real-time schema topology model. This dynamic approach keeps the
    # codebase dialect-resilient and lets the agent naturally discover schema changes.
    @traceable(name="Build Dynamic FK Map", run_type="tool")
    def _build_dynamic_fk_map(self) -> Dict[str, Dict[str, Any]]:
        """Fetches and maps foreign key constraints from PostgreSQL catalog tables.

        Returns:
            Dict[str, Dict[str, Any]]: Map containing all dynamic FK constraints.
        """
        logger.info("Building dynamic FK map from information_schema")
        # Direct catalog selection: Queries key constraint definitions from standard PostgreSQL information_schema.
        # Joins table_constraints with key_column_usage and constraint_column_usage to map source columns to target columns.
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
        fk_map: Dict[str, Dict[str, Any]] = {}
        try:
            # 1. Fetch psycopg active connection pool
            pool = self._get_pool()
            with pool.connection() as conn:
                # 2. Execute dynamic reflection catalog query
                results = conn.execute(query).fetchall()
                for row in results:
                    table = row["table_name"]
                    column = row["column_name"]
                    foreign_table = row["foreign_table_name"]
                    foreign_column = row["foreign_column_name"]
                    
                    # 3. Build nested topology structure: {source_table: {source_column: {target_table, target_column}}}
                    if table not in fk_map:
                        fk_map[table] = {}
                    fk_map[table][column] = {
                        "table": foreign_table,
                        "column": foreign_column
                    }
                return fk_map
        except Exception as e:
            logger.error(f"Failed to build dynamic FK map: {str(e)}", exc_info=True)
            return {}


    # ### --- [BREADTH-FIRST SCHEMA NAVIGATION] --- ###

    # [Elaborative Breakdown]
    # BFS Pathfinding for Relational Joins:
    # When a complex analytical query refers to tables that do not directly join (e.g., actor
    # and city), a naive LLM will either guess relationships or hallucinate intermediate keys.
    # We solve this by treating the database schema as an undirected graph, where tables are
    # nodes and foreign key constraints are edges.
    #
    # _find_shortest_path executes a classic Breadth-First Search (BFS) starting from an anchor
    # table, expanding outwards level-by-level to locate the target node. This guarantees
    # that the shortest relationship path is found, and that we discover all intermediate "bridge"
    # tables (e.g., film_actor, inventory, rental) required to safely execute multi-table joins.
    @traceable(name="Identify FK Bridge Tables", run_type="tool")
    def get_bridge_tables(self, anchors: List[str]) -> List[str]:
        """Identifies intermediate bridge tables required to join a list of anchor tables.

        Args:
            anchors: Active tables extracted from the user query.

        Returns:
            List[str]: Table names that act as intermediate bridges between anchors.
        """
        if len(anchors) < 2:
            return []
            
        bridges: Set[str] = set()
        # Set anchor table 0 as the starting search node for join-path calculation
        start_node: str = anchors[0]
        for target in anchors[1:]:
            # Retrieve shortest path (list of intermediate nodes) linking start_node to target
            path = self._find_shortest_path(start_node, target)
            if path:
                bridges.update(path)
        
        # Strip out original anchor tables to leave only the intermediate bridge tables
        return list(bridges - set(anchors))

    @traceable(name="BFS Pathfinding", run_type="tool")
    def _find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Performs a Breadth-First Search to find the shortest join path between two tables.

        Args:
            start: The table name representing the source node.
            end: The table name representing the target node.

        Returns:
            Optional[List[str]]: List of ordered table names from start to end, or None if unreachable.
        """
        # Node identity check: if start and end are identical, no pathfinding required.
        if start == end: 
            return [start]
            
        # Initialize BFS queue with starting node and path tracking list
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        # Maintain a visited set to avoid infinite cycles in the circular schema graph
        visited: Set[str] = {start}
        
        while queue:
            # Pop the first element from the queue (FIFO queue execution)
            (node, path) = queue.pop(0)
            
            # Explore all connected vertices (undirected neighbors in relationship graph)
            for next_node in self.graph.get(node, []):
                if next_node == end:
                    # Target node located! Return complete path
                    return path + [next_node]
                if next_node not in visited:
                    visited.add(next_node)
                    # Append new path segment to queue for further exploration
                    queue.append((next_node, path + [next_node]))
        return None


    # ### --- [METADATA & SCHEMA DISCOVERY] --- ###

    @traceable(name="Get Relevant FKs", run_type="tool")
    def get_relevant_fks(self, table_names: List[str]) -> List[Dict[str, str]]:
        """Maps specific foreign key relations active among a subset of tables.

        Args:
            table_names: The tables to discover relationship links for.

        Returns:
            List[Dict[str, str]]: List of structured links representing active joins.
        """
        all_fks = self.fk_map
        relevant_fks: List[Dict[str, str]] = []
        for table in table_names:
            if table in all_fks:
                # Iterate over active columns in the source table
                for col, target_info in all_fks[table].items():
                    # Filter: Only record relations if the target table is also active in our subset
                    if target_info["table"] in table_names:
                        relevant_fks.append({
                            "source_table": table,
                            "source_column": col,
                            "target_table": target_info["table"],
                            "target_column": target_info["column"]
                        })
        return relevant_fks

    def _get_pool(self) -> ConnectionPool:
        """Retrieves or lazy-initializes the psycopg connection pool for execution.

        Returns:
            ConnectionPool: The active engine connection pool.
        """
        if self._pool is None:
            # Initialize lazy pool for target database to keep resource utilization optimal
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
        """Executes a SQL statement securely and returns structured rows.

        Args:
            query: The compiled SQL text to run.

        Returns:
            Dict[str, Any]: Success payload containing data, or failure details.
        """
        logger.info(f"Executing SQL Query: {query}")
        try:
            # Open pool and contextually lease connection
            pool = self._get_pool()
            with pool.connection() as conn:
                # Run the statement, returning a structured list of dictionaries
                results = conn.execute(query).fetchall()
                logger.info(f"Query successful. Rows returned: {len(results)}")
                return {"status": "success", "data": results, "row_count": len(results)}
        except Exception as e:
            logger.error(f"SQL Execution Failed. Error: {str(e)}", exc_info=True)
            return {"status": "error", "error_message": str(e), "query": query}

    def _get_enum_map(self) -> Dict[str, List[str]]:
        """Queries database catalog to resolve custom ENUM labels and namespaces.

        Returns:
            Dict[str, List[str]]: Map of: {enum_type_name: [allowed_value_strings]}
        """
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
        enum_map: Dict[str, List[str]] = {}
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
            logger.error(f"Failed to fetch ENUM map: {str(e)}", exc_info=True)
            return {}

    def get_schema(self, table_names: Optional[List[str]] = None) -> str:
        """Builds a comprehensive, annotated schema description for LLM consumption.

        Automatically resolves custom ENUM labels, formats arrays with descriptive
        examples, and annotations for search-specific indexes (like tsvector).

        Args:
            table_names: Optional list of tables to restrict the schema description to.

        Returns:
            str: Annotated text schema block.
        """
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
                
                schema_parts: List[str] = []
                current_table: str = ""
                for col in columns:
                    if col["table_name"] != current_table:
                        current_table = col["table_name"]
                        schema_parts.append(f"\nTable: {current_table}")
                    
                    dtype = col["data_type"]
                    udt = col["udt_name"]
                    col_name = col["column_name"]
                    
                    # 1. Resolve ENUM types to let LLMs know literal options
                    if dtype == "USER-DEFINED" and udt in enum_map:
                        vals = ", ".join([f"'{v}'" for t in [udt] for v in enum_map[t]])
                        dtype = f"ENUM: {vals}"
                    
                    # 2. Add structural hints for ARRAY types
                    elif dtype == "ARRAY":
                        if col_name == "special_features":
                            dtype = "ARRAY (Examples: 'Trailers', 'Commentaries', 'Deleted Scenes', 'Behind the Scenes')"
                        else:
                            dtype = "ARRAY"
                            
                    # 3. Add explicit annotations for Full-Text Search columns
                    elif dtype == "tsvector":
                        dtype = "tsvector (Full-Text Search Index)"
                    
                    schema_parts.append(f" - {col_name} ({dtype})")
                
                return "\n".join(schema_parts)
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

    @traceable(name="Get Schema Object", run_type="tool")
    def get_schema_object(self, table_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Builds a fast structural dictionary of the tables and columns.

        Used during column pruning and security validation checks.

        Args:
            table_names: Specific tables to fetch.

        Returns:
            Dict[str, List[str]]: Mapping: {table_name: [list_of_columns]}
        """
        # Return cache if available and no specific tables requested
        if not table_names and self._schema_cache:
            return self._schema_cache

        pool = self._get_pool()
        query = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
        if table_names:
            tables_str = ",".join([f"'{t}'" for t in table_names])
            query += f" AND table_name IN ({tables_str})"
        
        schema_obj: Dict[str, List[str]] = {}
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
            logger.error(f"Error fetching schema object: {str(e)}", exc_info=True)
            return {}

    def list_tables(self) -> List[str]:
        """Lists all physical user tables in the public database schema.

        Returns:
            List[str]: Table names list.
        """
        pool = self._get_pool()
        try:
            with pool.connection() as conn:
                tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'").fetchall()
                return [t["table_name"] for t in tables]
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}", exc_info=True)
            return []

    def close(self) -> None:
        """Gracefully shuts down and purges the Psycopg connection pool."""
        if self._pool is not None:
            logger.info(f"Closing SQLEngine pool for: {self.db_name}")
            self._pool.close()
            self._pool = None


# ### --- [SINGLETON & SHUTDOWN CONFIGURATION] --- ###

# Globally available SQL Engine singleton
sql_engine: SQLEngine = SQLEngine()

# Register core connection pool for clean destruction at program end
atexit.register(sql_engine.close)
