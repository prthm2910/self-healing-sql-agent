from typing import Annotated
from pydantic import BaseModel, ValidationError
from langgraph.graph import StateGraph, START, END

# 1. Define a strict Pydantic State
class PydanticState(BaseModel):
    count: int  # Must be an integer!

def naughty_node(state: PydanticState):
    print(f"--- Inside Node: count is {type(state.count)} ---")
    # We are returning a STRING for an INTEGER field.
    # If LangGraph validated every step, this would crash.
    return {"count": "I am a string now!"}

# 2. Build the graph
builder = StateGraph(PydanticState)
builder.add_node("naughty", naughty_node)
builder.add_edge(START, "naughty")
builder.add_edge("naughty", END)
graph = builder.compile()

# TEST 1: Entry Validation
print("TEST 1: Starting with invalid data...")
try:
    graph.invoke({"count": "not an int"})
    print("FAILED: Entry validation didn't catch the bad data!")
except ValidationError as e:
    print("SUCCESS: Pydantic blocked the invalid entry!")

# TEST 2: Mid-Graph Corruption
print("\nTEST 2: Starting with valid data, but corrupting it inside a node...")
try:
    result = graph.invoke({"count": 1})
    print(f"Graph Result: {result}")
    print(f"Final count type: {type(result['count'])}")
    if isinstance(result['count'], str):
        print("OBSERVATION: The graph finished with a STRING in an INTEGER field! Validation failed mid-graph.")
except Exception as e:
    print(f"CRASHED: {e}")
