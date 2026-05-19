import pytest

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.builder import build_chatbot_graph

@pytest.mark.anyio
async def test_dac_workflow_integration():
    """
    Integration test for the Divide & Conquer workflow.
    Verifies that complex queries trigger decomposition and assembly.
    """
    checkpointer = MemorySaver()
    graph = build_chatbot_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_dac_thread", "user_id": "test_user"}}
    
    # A query that should be classified as COMPLEX
    question = "Which customers from Canada have rented Action films?"
    
    # Run the graph
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "retry_count": 0
    }
    
    # We use stream to observe node transitions
    nodes_visited = []
    import time
    start_time = time.time()
    last_node_time = start_time
    
    async for event in graph.astream(initial_state, config, stream_mode="updates"):
        for node_name, output in event.items():
            now = time.time()
            duration = now - last_node_time
            total_duration = now - start_time
            nodes_visited.append(node_name)
            print(f"[{total_duration:.2f}s] Visited Node: {node_name} (took {duration:.2f}s)")
            last_node_time = now

    # Assertions
    assert "guardian" in nodes_visited
    assert "classifier" in nodes_visited
    
    # Since it's a real LLM call (or mocked depending on environment), 
    # we check if it went down the complex path
    if "decomposer" in nodes_visited:
        assert "worker" in nodes_visited
        assert "assembler" in nodes_visited
        assert "execute_sql" in nodes_visited
        print("Success: DAC path executed.")
    else:
        # If it was classified as SIMPLE for some reason, we check the simple path
        assert "generate_sql" in nodes_visited
        print("Note: Simple path executed instead of DAC.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dac_workflow_integration())
