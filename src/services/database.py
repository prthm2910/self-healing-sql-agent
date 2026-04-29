import os

import psycopg
from psycopg_pool import ConnectionPool

from src.core.config import settings

# Global Connection Pool with Health Checks
DB_POOL = ConnectionPool(
    conninfo=settings.database_url, 
    max_size=20,
    min_size=1, # Keep at least one connection warm
    check=ConnectionPool.check_connection, # PRO FIX: Verify health before use
    kwargs={"autocommit": True}
)


def get_connection_pool():
    """Returns the global connection pool."""
    return DB_POOL


def delete_thread_data(user_id: str, thread_id: str, store):
    """Purges thread metadata and history using the provided store and direct SQL."""
    store.delete((user_id, "threads"), thread_id)
    with DB_POOL.connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
