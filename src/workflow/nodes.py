from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.services.llm import get_chat_model
from src.core.config import settings
from src.prompts.assistant import get_assistant_prompt


def call_chatbot(state: State, config: RunnableConfig, store=None):
    """The node that retrieves memories and invokes the LLM."""
    llm = get_chat_model()
    user_id = config.get("configurable", {}).get("user_id", settings.default_user_id)
    
    # 1. RAG: Search long-term memory
    query = state["messages"][-1].content if state["messages"] else ""
    
    if store is not None:
        # 1a. Try semantic search first
        if query:
            memories = store.search(
                (user_id, "memories"),
                query=query,
                limit=5
            )
        else:
            memories = []
            
        # 1b. Fallback: If no semantic matches, get the most recent facts for this user
        if not memories:
            recent_items = store.search(
                (user_id, "memories"),
                query=None,
                limit=5
            )
            memories = recent_items
            if memories:
                print(f"User {user_id}: Found {len(memories)} recent memories (fallback from search)")
        else:
            print(f"User {user_id}: Found {len(memories)} semantic memories for query '{query}'")
    else:
        print(f"Warning: Store is None for User {user_id}")
        memories = []
    
    formatted_memories = "\n".join(
        [f"- {m.value['fact']}" for m in memories]
    ) if memories else "No previous memories found."
    
    # 2. Get Prompt Template from Factory
    prompt_template = get_assistant_prompt()
    
    # 3. Sliding Window (last 20 messages)
    context_window = state["messages"][-settings.context_window_size:] if len(state["messages"]) > settings.context_window_size else state["messages"]
    
    # 4. Format and Invoke
    chain = prompt_template | llm
    response = chain.invoke({
        "memories": formatted_memories,
        "tag": settings.memory_tag,
        "messages": context_window
    })
    
    return {"messages": [response]}
