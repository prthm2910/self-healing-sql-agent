import os
from typing import List, Dict, Any, Optional

import atexit
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv

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
        self.connection_url = base_url.rsplit("/", 1)[0] + f"/{db_name}"
        
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

    def get_schema(self, table_names: Optional[List[str]] = None) -> str:
        """Returns schema using the lazy-loaded pool."""
        pool = self._get_pool()
        query = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"
        if table_names:
            tables_str = ",".join([f"'{t}'" for t in table_names])
            query += f" AND table_name IN ({tables_str})"
        
        try:
            with pool.connection() as conn:
                columns = conn.execute(query).fetchall()
                
                schema_parts = []
                current_table = ""
                for col in columns:
                    if col["table_name"] != current_table:
                        current_table = col["table_name"]
                        schema_parts.append(f"\nTable: {current_table}")
                    
                    schema_parts.append(f" - {col['column_name']} ({col['data_type']})")
                
                return "\n".join(schema_parts)
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

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
