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

def test_memory():
    pool = get_connection_pool()
    store = PostgresStore(
        pool,
        index={
            "dims": settings.embedding_dimensions,
            "embed": get_embeddings_provider(),
            "fields": ["$"]
        }
    )
    store.setup()
    
    user_id = "test_user"
    namespace = (user_id, "memories")
    
    # Put a test fact
    fact_id = "test_fact_1"
    fact_data = {"fact": "The user likes coffee.", "category": "preference", "certainty": 0.9}
    
    print(f"Putting fact: {fact_data}")
    store.put(namespace, fact_id, fact_data)
    
    # Search for it
    query = "What does the user like?"
    print(f"Searching for: '{query}' in namespace {namespace}")
    results = store.search(namespace, query=query, limit=5)
    
    print(f"Found {len(results)} results.")
    for r in results:
        print(f"- {r.value}")

if __name__ == "__main__":
    test_memory()
