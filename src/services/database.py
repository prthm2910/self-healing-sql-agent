import os
import psycopg
from psycopg_pool import ConnectionPool
from src.core.config import settings
from src.utils.logger import logger

_DB_POOL = None

def get_connection_pool():
    """
    Returns the global connection pool, initializing it only once (Lazy Loading).
    """
    global _DB_POOL
    if _DB_POOL is None:
        logger.info("Initializing database connection pool...")
        try:
            _DB_POOL = ConnectionPool(
                conninfo=settings.database_url, 
                max_size=2, 
                min_size=1,
                timeout=60.0,
                check=ConnectionPool.check_connection,
                kwargs={"autocommit": True}
            )
            logger.info("Database connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}", exc_info=True)
            raise
    return _DB_POOL

def delete_thread_data(user_id: str, thread_id: str, store):
    """Purges thread metadata and history."""
    logger.info(f"Deleting thread data for User: {user_id}, Thread: {thread_id}")
    try:
        pool = get_connection_pool()
        store.delete((user_id, "threads"), thread_id)
        with pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        logger.info(f"Successfully deleted thread data for Thread: {thread_id}")
    except Exception as e:
        logger.error(f"Error deleting thread data for Thread {thread_id}: {e}", exc_info=True)
        raise
