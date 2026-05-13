import asyncio
from langchain_core.messages import HumanMessage
from src.workflow.nodes import decomposer_node
from src.workflow.state import State
from src.core.config import settings

async def debug_decomposer():
    print("Testing decomposer_node in isolation...")
    state = {
        "messages": [HumanMessage(content="Which customers from Canada have rented Action films?")],
        "selected_tables": ["customer", "address", "city", "country", "rental", "inventory", "film", "film_category", "category"],
        "fk_relationships": [
            {"source_table": "customer", "source_column": "address_id", "target_table": "address", "target_column": "address_id"},
            {"source_table": "address", "source_column": "city_id", "target_table": "city", "target_column": "city_id"},
            {"source_table": "city", "source_column": "country_id", "target_table": "country", "target_column": "country_id"}
        ],
        "agent_logs": []
    }
    
    config = {"configurable": {"user_id": "debug_user", "thread_id": "debug_thread"}}
    
    try:
        print("Invoking decomposer_node...")
        # decomposer_node is sync, but we call it in a way that mimics the graph
        result = decomposer_node(state, config)
        print("Success!")
        print(f"Tasks: {len(result['sub_tasks'])}")
        print(f"Join Plan: {result['join_plan']}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_decomposer())
