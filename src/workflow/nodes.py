from langchain_core.runnables import RunnableConfig
from src.services.llm import get_chat_model
from src.services.sql_engine import sql_engine
from src.prompts.sql_agent import get_sql_generation_prompt, get_sql_healing_prompt
from src.prompts.assistant import get_assistant_prompt
from src.core.config import settings
from src.workflow.state import State
from langchain_core.messages import AIMessage, HumanMessage

# Initialize models
chat_model = get_chat_model()

def call_chatbot(state: State, config: RunnableConfig, store=None):
    """
    Standard chatbot node with RAG (Memory) capabilities.
    """
    user_id = config.get("configurable", {}).get("user_id", settings.default_user_id)
    query = state["messages"][-1].content if state["messages"] else ""

    # 1. RAG: Search long-term memory
    if store is not None:
        memories = store.search((user_id, "memories"), query=query, limit=5)
        if not memories:
            memories = store.search((user_id, "memories"), query=None, limit=5)
    else:
        memories = []

    formatted_memories = "\n".join([f"- {m.value['fact']}" for m in memories]) if memories else "No previous memories found."

    # 2. Get Prompt and Apply Sliding Window
    prompt_template = get_assistant_prompt()
    window_size = getattr(settings, "context_window_size", 20)
    context_window = state["messages"][-window_size:] if len(state["messages"]) > window_size else state["messages"]

    # 3. Invoke
    chain = prompt_template | chat_model
    response = chain.invoke({
        "memories": formatted_memories,
        "tag": settings.memory_tag,
        "messages": context_window
    })

    return {"messages": [response]}

def generate_sql_node(state: State):
    """
    Analyzes the user request and generates an initial SQL query.
    """
    schema = sql_engine.get_schema()
    prompt_template = get_sql_generation_prompt()
    user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    
    chain = prompt_template | chat_model
    response = chain.invoke({
        "schema": schema,
        "history": state["messages"][:-1],
        "question": user_question
    })
    
    sql_query = response.content.strip().replace("```sql", "").replace("```", "")
    return {"current_sql": sql_query, "retry_count": 0}

def execute_sql_node(state: State):
    """
    Executes the current SQL query and captures results or errors.
    """
    print(f"Executing SQL: {state['current_sql']}")
    result = sql_engine.execute_query(state["current_sql"])
    
    if result["status"] == "success":
        return {"sql_results": result["data"], "sql_error": None}
    else:
        return {"sql_error": result["error_message"], "sql_results": []}

def heal_sql_node(state: State):
    """
    Attempts to fix a failing SQL query using the error message.
    """
    schema = sql_engine.get_schema()
    prompt_template = get_sql_healing_prompt()
    user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

    chain = prompt_template | chat_model
    response = chain.invoke({
        "schema": schema,
        "failed_query": state["current_sql"],
        "error_message": state["sql_error"],
        "question": user_question
    })
    
    fixed_sql = response.content.strip().replace("```sql", "").replace("```", "")
    return {"current_sql": fixed_sql, "retry_count": state["retry_count"] + 1}

def format_sql_response_node(state: State):
    """
    Formats the final SQL results into a natural language message.
    """
    if not state["sql_results"]:
        content = f"I ran the query but found no results.\nSQL: `{state['current_sql']}`"
    else:
        data_str = str(state["sql_results"][:5]) 
        content = f"Here is what I found in the database:\n{data_str}\n\n(Total rows: {len(state['sql_results'])})"
        
    return {"messages": [AIMessage(content=content)]}
