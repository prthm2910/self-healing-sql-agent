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

def test_multified_index():
    pool = get_connection_pool()
    
    # Try an index that includes lesson fields
    store = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["fact", "instruction", "title", "mistake"]
        }
    )
    store.setup()
    
    namespace = ("global", "lessons", "dynamic")
    
    # Put a lesson
    lesson_id = "test_lesson_index"
    lesson_data = {
        "title": "Vegetarian Filter",
        "mistake": "Suggested meat toppings.",
        "instruction": "When the user mentions pizza, suggest non-meat toppings.",
        "reasoning": "User is a vegetarian."
    }
    store.put(namespace, lesson_id, lesson_data)
    
    # Search for it
    query = "What pizza toppings?"
    print(f"Searching for: '{query}' in namespace {namespace}")
    results = store.search(namespace, query=query, limit=5)
    
    print(f"Found {len(results)} results.")
    for r in results:
        print(f"- {r.value.get('title')}: {r.value.get('instruction')}")

if __name__ == "__main__":
    test_multified_index()
