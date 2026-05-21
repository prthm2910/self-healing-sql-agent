import uuid

from langchain_core.messages import HumanMessage

from src.core.config import settings
from src.workflow.builder import build_chatbot_graph

def test_query():
    graph = build_chatbot_graph()
    thread_id = str(uuid.uuid4())
    user_id = "user_123"
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    
    question = "List the top 5 customers by total payment amount, including their full name, city, and country."
    print(f"\n--- Testing Query: {question} ---\n")
    
    input_data = {
        "messages": [HumanMessage(content=question)],
        "user_id": user_id
    }
    
    try:
        response = graph.invoke(input_data, config=config)
        print("\n--- FINAL RESPONSE ---\n")
        print(response["messages"][-1].content)
        print("\n--- END ---\n")
    except Exception as e:
        print(f"\n--- ERROR ---\n{e}")

if __name__ == "__main__":
    test_query()
