from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.workflow.state import State
from src.services.llm import get_chat_model
from src.utils.logger import logger
from src.utils.limiter import rate_limiter

guardian_model = get_chat_model(is_flash=True)

def guardian_node(state: State):
    """
    Orchestration Node: Decides the intent and security of the message.
    """
    logger.info("Node: guardian_node")
    
    # 1. Enforcement: Global Rate Limiting
    if not rate_limiter.check_and_record():
        logger.warning("Global Rate Limit Reached!")
        return {
            "intent": "blocked",
            "messages": [AIMessage(content="⚠️ System busy. Please wait a moment.")]
        }
    
    last_msg = state["messages"][-1].content
    
    # 2. Intent & Safety Analysis
    decision_prompt = f"""You are an Intent Classifier and Security Guardian.
Analyze the user message and categorize it into EXACTLY one of these intents:

1. SQL: The user is asking for information that requires searching a database (e.g., questions about movies, actors, rentals, inventory, etc.).
2. BLOCKED: The user is explicitly asking to DELETE, FORGET, or REMOVE their personal facts or history.
3. CHAT: General greeting, personal preference sharing, or any non-database question.

User Message: "{last_msg}"

Rules:
- If intent is BLOCKED, output: "INTENT: BLOCKED"
- If intent is SQL, output: "INTENT: SQL"
- Otherwise, output: "INTENT: CHAT"

ONLY output the tag. No explanation.
"""
    
    res = guardian_model.invoke([SystemMessage(content=decision_prompt)]).content.strip().upper()
    
    if "BLOCKED" in res:
        logger.info("Guardian detected BLOCKED intent.")
        return {
            "intent": "blocked",
            "messages": [AIMessage(content="🛑 For your security, memory deletion can only be performed manually via the 'Long-Term Memory' manager in the sidebar.")]
        }
    elif "SQL" in res:
        logger.info("Guardian detected SQL intent.")
        return {"intent": "sql"}
    else:
        logger.info("Guardian detected CHAT intent.")
        return {"intent": "chat"}
