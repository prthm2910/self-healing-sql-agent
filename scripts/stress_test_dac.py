import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.workflow.builder import build_chatbot_graph
from src.utils.logger import logger

async def run_stress_test():
    """
    Executes the 'Join Marathon' query to stress test the DAC architecture.
    """
    checkpointer = MemorySaver()
    graph = build_chatbot_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "stress_test_1", "user_id": "test_user"}}
    
    # The Join Marathon Query
    question = "List the top 3 actors whose films have generated the most revenue from customers living in cities that start with the letter 'A', but only for 'Sci-Fi' films."
    
    print(f"\n--- STARTING STRESS TEST ---")
    print(f"Question: {question}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "retry_count": 0
    }
    
    async for event in graph.astream(initial_state, config, stream_mode="updates"):
        for node_name, output in event.items():
            print(f"DEBUG: Node Finished -> {node_name}")
            
            # Check for specific DAC triggers
            if node_name == "classifier":
                print(f"  > Classification: {'COMPLEX' if output.get('is_complex') else 'SIMPLE'}")
            
            if node_name == "decomposer":
                tasks = output.get("sub_tasks", [])
                print(f"  > Decomposition: {len(tasks)} sub-tasks created.")
                for i, t in enumerate(tasks):
                    print(f"    {i+1}. {t.get('description')}")
            
            if node_name == "assembler":
                sql = output.get("current_sql", "")
                print(f"  > Assembly Complete. SQL Length: {len(sql)}")

            if node_name == "format_sql_response":
                print("\n--- FINAL RESPONSE ---")
                msg = output.get("messages", [None])[0]
                if msg:
                    print(msg.content)

if __name__ == "__main__":
    asyncio.run(run_stress_test())
