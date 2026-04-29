import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

from src.utils.logger import logger

class SQLEngine:
    def __init__(self, db_name: str = "pagila"):
        base_url = os.getenv("DATABASE_URL")
        if not base_url:
            logger.error("DATABASE_URL not found in environment")
            raise ValueError("DATABASE_URL not found in environment")
        
        # Replace default db with target db
        self.connection_url = base_url.rsplit("/", 1)[0] + f"/{db_name}"
        logger.debug(f"SQLEngine initialized for database: {db_name}")
        
    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Executes a SQL query and returns results or a detailed error.
        """
        logger.info(f"Executing SQL Query: {query}")
        try:
            with psycopg.connect(self.connection_url, row_factory=dict_row) as conn:
                results = conn.execute(query).fetchall()
                logger.info(f"Query successful. Rows returned: {len(results)}")
                return {
                    "status": "success",
                    "data": results,
                    "row_count": len(results)
                }
        except Exception as e:
            logger.error(f"SQL Execution Failed. Error: {str(e)} | Query: {query}")
            return {
                "status": "error",
                "error_message": str(e),
                "query": query
            }

    def get_schema(self, table_names: Optional[List[str]] = None) -> str:
        """
        Returns the DDL/Schema for the requested tables.
        """
        logger.debug(f"Retrieving schema for tables: {table_names if table_names else 'ALL'}")
        query = """
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public'
        """
        if table_names:
            tables_str = ",".join([f"'{t}'" for t in table_names])
            query += f" AND table_name IN ({tables_str})"
        
        query += " ORDER BY table_name, ordinal_position"
        
        try:
            with psycopg.connect(self.connection_url, row_factory=dict_row) as conn:
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
        """
        Lists all available tables in the public schema.
        """
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        try:
            with psycopg.connect(self.connection_url) as conn:
                tables = conn.execute(query).fetchall()
                return [t[0] for t in tables]
        except Exception as e:
            print(f"Error listing tables: {e}")
            return []

# Singleton instance for the Pagila DB
sql_engine = SQLEngine()
