from src.workflow.builder import sql_chatbot_graph
from langchain_core.messages import HumanMessage
import uuid

def test_unified_agent():
    print("Testing Unified AI Assistant (Chat + SQL)...")
    
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": "test_user"}}
    
    # Test 1: Standard Chat
    print("\n--- Test 1: Standard Chat ---")
    inputs = {"messages": [HumanMessage(content="Hello! Can you remember that my favorite color is blue?")]}
    for event in sql_chatbot_graph.stream(inputs, config=config):
        for node, _ in event.items():
            print(f"Executed Node: {node}")

    # Test 2: SQL Query (Triggered by keywords)
    print("\n--- Test 2: SQL Query ---")
    inputs = {"messages": [HumanMessage(content="How many films are in the database?")]}
    for event in sql_chatbot_graph.stream(inputs, config=config):
        for node, data in event.items():
            print(f"Executed Node: {node}")
            if "current_sql" in data:
                print(f"  Generated SQL: {data['current_sql']}")

if __name__ == "__main__":
    test_unified_agent()
