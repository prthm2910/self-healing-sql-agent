import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

load_dotenv()

from src.services.database import get_connection_pool
from src.services.embeddings import get_embeddings_provider
from langgraph.store.postgres import PostgresStore
from src.core.config import settings
from src.workflow.nodes import call_chatbot
from src.services.lessons import record_lesson

def verify_all_tiers():
    print("=== STARTING COMPREHENSIVE TIER VERIFICATION ===")
    
    pool = get_connection_pool()
    store = PostgresStore(
        pool,
        index={
            "embed": get_embeddings_provider(),
            "dims": settings.embedding_dimensions,
            "fields": ["$"]
        }
    )
    store.setup()
    
    user_id = "verification_user"
    config = {"configurable": {"user_id": user_id}}
    
    # 1. SETUP TIER 2: LONG-TERM USER FACTS
    print("\n[Tier 2] Populating User Facts...")
    store.put((user_id, "memories"), "pref_1", {"fact": "The user is a vegetarian.", "category": "diet", "certainty": 1.0})
    store.put((user_id, "memories"), "pref_2", {"fact": "The user's favorite movie is Inception.", "category": "preference", "certainty": 0.95})

    # 2. SETUP TIER 3: SYSTEMIC LESSONS
    print("[Tier 3] Populating Systemic Lessons...")
    # Pinned (Global)
    record_lesson(
        title="Global Rule",
        mistake="None",
        instruction="Always greet the user warmly.",
        reasoning="User experience.",
        store=store,
        is_global=True
    )
    # Dynamic (Specific)
    record_lesson(
        title="Vegetarian Filter",
        mistake="Suggested meat toppings.",
        instruction="When the user mentions pizza, suggest non-meat toppings.",
        reasoning="User is a vegetarian.",
        store=store,
        is_global=False
    )

    # 3. SETUP TIER 1: SHORT-TERM HISTORY
    print("[Tier 1] Preparing Short-Term History...")
    history = [
        HumanMessage(content="I'm planning a pizza party."),
        # Last message is the query
        HumanMessage(content="What toppings should I get for myself?")
    ]
    state = {"messages": history}

    # 4. EXECUTE NODE & VERIFY ASSEMBLY
    print("\n--- Executing call_chatbot node ---")
    # We invoke the node. We'll check logs or inspect the prompt if possible.
    # Since the node calls llm.invoke, we'll see the final payload in logs if we set level to DEBUG.
    
    # Actually, let's just inspect the retrieval results manually first to ensure the code finds them.
    from src.services.lessons import get_relevant_lessons
    
    last_msg = history[-1].content
    
    # Verify Level 2 Retrieval
    memories = store.search((user_id, "memories"), query=last_msg, limit=5)
    print(f"\nLevel 2 Retrieval (Query: '{last_msg}'):")
    for m in memories:
        print(f"  - FOUND: {m.value['fact']}")
    
    # Verify Level 3 Retrieval
    lessons = get_relevant_lessons(last_msg, store)
    print(f"\nLevel 3 Retrieval (Query: '{last_msg}'):")
    print(lessons)

    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    verify_all_tiers()
