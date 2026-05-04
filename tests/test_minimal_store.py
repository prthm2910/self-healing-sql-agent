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

def test_minimal_store():
    pool = get_connection_pool()
    
    # Standard index
    store = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["fact"]
        }
    )
    store.setup()
    
    namespace = ("test", "minimal")
    store.put(namespace, "item1", {"fact": "The sky is blue."})
    
    results = store.search(namespace, query="What color is the sky?", limit=1)
    print(f"Standard Search Results: {len(results)}")
    if results:
        print(f"  - {results[0].value}")

if __name__ == "__main__":
    test_minimal_store()
