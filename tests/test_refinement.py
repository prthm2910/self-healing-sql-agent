import asyncio
import os
from langchain_core.messages import HumanMessage
from src.workflow.builder import build_chatbot_graph
from src.core.config import settings

async def main():
    graph = build_chatbot_graph()
    config = {"configurable": {"thread_id": "test_thread", "user_id": "test_user"}}
    
    # 1. Irrelevant Question
    print("\n--- TEST 1: Irrelevant Question ---")
    input_data = {"messages": [HumanMessage(content="What's the weather in London?")], "user_id": "test_user"}
    response = graph.invoke(input_data, config=config)
    print(f"Intent: {response.get('intent')}")
    print(f"Response: {response['messages'][-1].content}")
    
    # 2. SQL Question (check if it tries to store memory - difficult to check tag here but can see response)
    print("\n--- TEST 2: SQL Question ---")
    input_data = {"messages": [HumanMessage(content="Who are the top 5 actors?")], "user_id": "test_user"}
    response = graph.invoke(input_data, config=config)
    print(f"Intent: {response.get('intent')}")
    # print(f"Response: {response['messages'][-1].content}") # Might be long
    print(f"Has Memory Tag: {settings.memory_tag in response['messages'][-1].content}")

    # 3. Personal Fact
    print("\n--- TEST 3: Personal Fact ---")
    input_data = {"messages": [HumanMessage(content="I am from New York")], "user_id": "test_user"}
    response = graph.invoke(input_data, config=config)
    print(f"Intent: {response.get('intent')}")
    print(f"Response: {response['messages'][-1].content}")
    print(f"Has Memory Tag: {settings.memory_tag in response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
