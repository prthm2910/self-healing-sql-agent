import os
import sys
from dotenv import load_dotenv

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

load_dotenv()

from src.services.database import get_connection_pool
from src.services.embeddings import get_embeddings_provider
from langgraph.store.postgres import PostgresStore
from src.core.config import settings

def test_search_no_query():
    pool = get_connection_pool()
    store = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["fact"]
        }
    )
    store.setup()
    
    user_id = "test_user"
    namespace = (user_id, "memories")
    
    print(f"Searching with query=None in namespace {namespace}")
    try:
        results = store.search(namespace, query=None, limit=5)
        print(f"Found {len(results)} results with query=None.")
    except Exception as e:
        print(f"Search with query=None FAILED: {e}")

    print(f"Searching with query='' in namespace {namespace}")
    try:
        results = store.search(namespace, query="", limit=5)
        print(f"Found {len(results)} results with query=''.")
    except Exception as e:
        print(f"Search with query='' FAILED: {e}")

if __name__ == "__main__":
    test_search_no_query()
