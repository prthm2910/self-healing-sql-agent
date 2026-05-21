# ### --- [IMPORTS & CONFIGURATION] --- ###
import os
import threading
from typing import Any, Optional

import psycopg
from psycopg_pool import ConnectionPool

from src.core.config import settings
from src.utils.logger import logger


# ### --- [STATE DEFINITION] --- ###

# [Elaborative Breakdown]
# Threading Safety & Global State:
# We maintain a single global database connection pool (_DB_POOL) shared across all
# worker threads in the LangGraph runtime. To prevent race conditions and redundant
# connection pool instantiations when multiple concurrent queries arrive, we utilize
# a standard threading.Lock. This lock acts as a synchronization barrier, ensuring
# that only a single execution thread initializes the pool, while subsequent threads
# yield and reuse the existing instance.
_DB_POOL: Optional[ConnectionPool] = None
_POOL_LOCK: threading.Lock = threading.Lock()


# ### --- [CONNECTION INITIALIZATION] --- ###

def get_connection_pool() -> ConnectionPool:
    """Retrieves or lazy-initializes the global PostgreSQL connection pool.

    Uses a thread-safe Double-Checked Locking (DCL) pattern to ensure that the
    ConnectionPool is instantiated exactly once, eliminating lock contention on subsequent
    reads after the pool has been established.

    Returns:
        ConnectionPool: The active, thread-safe global connection pool.

    Raises:
        psycopg.OperationalError: If initialization fails due to invalid credentials,
            network timeouts, or database server unreachable states.
    """
    global _DB_POOL
    # FIRST CHECK (Non-blocking): Avoid acquiring lock overhead if the pool is already initialized.
    if _DB_POOL is None:
        # [Elaborative Breakdown]
        # Double-Checked Locking Pattern (DCL):
        # We check '_DB_POOL is None' twice. The first check is non-blocking (read)
        # to avoid lock acquisition overhead once initialization completes. The second
        # check is executed inside the critical section (under '_POOL_LOCK') to ensure
        # that if two threads bypassed the first check concurrently, only the first one
        # inside the lock actually instantiates the pool. The second thread will see that
        # '_DB_POOL' is no longer None and safely bypass instantiation.
        with _POOL_LOCK:
            # SECOND CHECK (Blocking critical section): Guard against concurrent instantiation.
            if _DB_POOL is None:
                logger.info("Initializing database connection pool...")
                try:
                    # Instantiate ConnectionPool with optimized parameters:
                    # - max_size=2: Enforces conservative pool capping to prevent connection starvation on Neon free tiers.
                    # - min_size=1: Ensures at least one pre-warmed connection is warm for fast initial request times.
                    # - timeout=60.0: Generous timeout margin for server cold starts.
                    # - check=ConnectionPool.check_connection: Active health validation to automatically prune dead sockets.
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


# ### --- [DATA PURGING SERVICES] --- ###

def delete_thread_data(user_id: str, thread_id: str, store: Any) -> None:
    """Purges checkpoints, blobs, and metadata associated with a specific graph thread.

    This ensures state hygiene by cleaning up the persistence layer when a thread
    reaches its terminal execution or is manually reset by the user.

    Args:
        user_id: The unique identifier of the user requesting the purge.
        thread_id: The specific thread ID whose history needs deletion.
        store: The state store checkpoint adapter used by LangGraph.

    Raises:
        psycopg.DatabaseError: If delete operations fail or transaction commit is aborted.
    """
    logger.info(f"Deleting thread data for User: {user_id}, Thread: {thread_id}")
    try:
        # Retrieve the shared pool instance to open connection
        pool = get_connection_pool()
        
        # Purge thread-scoped document/blob stores managed by LangGraph store first
        store.delete((user_id, "threads"), thread_id)
        
        # Acquire a single connection from the pool and run the deletes inside a single ACID transaction
        with pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Enforce sequential deletion honoring structural foreign key dependency hierarchies:
                    # We clear writes, blobs, and then checkpoints to avoid constraint violations.
                    logger.debug(f"Purging checkpoints table references for thread: {thread_id}")
                    cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                    
                    logger.debug(f"Purging checkpoint_blobs references for thread: {thread_id}")
                    cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    
                    logger.debug(f"Purging checkpoint_writes references for thread: {thread_id}")
                    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        logger.info(f"Successfully deleted thread data for Thread: {thread_id}")
    except Exception as e:
        logger.error(f"Error deleting thread data for Thread {thread_id}: {e}", exc_info=True)
        raise
