import os
import sys
import time
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from src.workflow.state import State
from src.workflow.nodes import anchor_selector_node, column_pruner_node
from src.utils.logger import logger
import logging

# Set logging to INFO for verbosity
logging.getLogger("ai_assistant").setLevel(logging.INFO)

def run_interactive_test():
    print("="*60)
    print("🚀 AGGRESSIVE SCHEMA PRUNING TESTER")
    print("="*60)
    print("Enter 'exit' to quit.\n")

    config = RunnableConfig(configurable={"user_id": "interactive_tester", "thread_id": "test_session"})

    while True:
        query = input("\n🔍 Enter Test Query: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        if not query:
            continue

        state = State(
            messages=[HumanMessage(content=query)],
            agent_logs=[],
            selected_tables=[],
            selected_columns={},
            fk_relationships=[]
        )

        print("\n" + "-"*40)
        print("🛠️ PHASE 1: ANCHOR SELECTION & BRIDGE TRAVERSAL")
        print("-"*40)
        
        start_p1 = time.time()
        try:
            res1 = anchor_selector_node(state, config)
            state.update(res1)
            duration1 = time.time() - start_p1
            
            log = res1["agent_logs"][-1]
            print(f"✅ Status: SUCCESS ({duration1:.2f}s)")
            print(f"📍 Anchors: {log.get('anchors')}")
            print(f"🌉 Bridges: {log.get('bridges')}")
            print(f"📦 Total Tables: {res1['selected_tables']}")
        except Exception as e:
            print(f"❌ Phase 1 Failed: {e}")
            continue

        print("\n" + "-"*40)
        print("✂️ PHASE 2: SURGICAL COLUMN PRUNING")
        print("-"*40)
        
        start_p2 = time.time()
        try:
            res2 = column_pruner_node(state, config)
            state.update(res2)
            duration2 = time.time() - start_p2
            
            print(f"✅ Status: SUCCESS ({duration2:.2f}s)")
            print(f"🔗 FKs Preserved: {len(res2.get('fk_relationships', []))}")
            
            print("\n📂 PRUNED SCHEMA MAP:")
            for table, cols in res2["selected_columns"].items():
                print(f"  • {table: <15} -> {cols}")
                
        except Exception as e:
            print(f"❌ Phase 2 Failed: {e}")
            continue

        print("\n" + "="*60)
        print(f"⏱️ Total Latency: {time.time() - start_p1:.2f}s")
        print("="*60)

if __name__ == "__main__":
    run_interactive_test()
