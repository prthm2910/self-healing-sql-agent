import os
import sys
import psycopg
from dotenv import load_dotenv

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

load_dotenv()

from src.services.database import get_connection_pool
from src.services.embeddings import get_embeddings_provider
from langgraph.store.postgres import PostgresStore
from src.core.config import settings

def simple_fix_and_test():
    db_url = os.getenv("DATABASE_URL")
    print("🗑️  Dropping store table...")
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS store CASCADE;")
                conn.commit()
    except Exception as e:
        print(f"Drop failed: {e}")

    # Use a direct connection for setup to be absolutely sure
    print("🛠️  Setting up store using direct connection...")
    try:
        with psycopg.connect(db_url) as conn:
            # We need to pass the conn to PostgresStore, but it expects a pool or conn.
            # In LangGraph, it can take a connection.
            index_config = {
                "embed": get_embeddings_provider(),
                "dims": settings.embedding_dimensions,
                "fields": ["$"] # Index everything
            }
            store = PostgresStore(conn, index=index_config)
            store.setup()
            conn.commit()
            print("✅ Store setup successful via direct connection.")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return

    # Now use the pool for operations
    pool = get_connection_pool()
    store = PostgresStore(pool, index=index_config)

    # Test Put 1: Memory
    print("\nTesting Memory Put...")
    try:
        store.put(("user_1", "memories"), "mem_1", {"fact": "User likes blue."})
        print("✅ Memory Put successful.")
    except Exception as e:
        print(f"❌ Memory Put failed: {e}")

    # Test Put 2: Lesson
    print("\nTesting Lesson Put...")
    try:
        store.put(("global", "lessons", "dynamic"), "less_1", {
            "title": "SQL Error",
            "mistake": "Wrong table name.",
            "instruction": "Use table 'rental' not 'rentals'.",
            "reasoning": "Schema consistency."
        })
        print("✅ Lesson Put successful.")
    except Exception as e:
        print(f"❌ Lesson Put failed: {e}")

    # Test Search 1: Memory
    print("\nTesting Memory Search...")
    res = store.search(("user_1", "memories"), query="What color?", limit=1)
    if res:
        print(f"✅ Found: {res[0].value['fact']}")
    else:
        print("❓ No memory found.")

    # Test Search 2: Lesson
    print("\nTesting Lesson Search...")
    res = store.search(("global", "lessons", "dynamic"), query="rental table", limit=1)
    if res:
        print(f"✅ Found: {res[0].value['instruction']}")
    else:
        print("❓ No lesson found.")

if __name__ == "__main__":
    simple_fix_and_test()
