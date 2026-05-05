from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.services.llm import get_chat_model
from src.services.sql_engine import sql_engine
from src.prompts.sql_agent import (
    get_sql_generation_prompt, 
    get_sql_healing_prompt, 
    get_sql_response_format_prompt
)
from src.prompts.assistant import get_assistant_prompt
from src.core.config import settings
from src.workflow.state import State
from src.workflow.schema import SQLResponse
from src.utils.logger import logger, log_context
from src.utils.table import generate_markdown_table
from src.utils.limiter import rate_limiter
from src.services.lessons import get_relevant_lessons, record_lesson

# Initialize models
chat_model = get_chat_model()
guardian_model = get_chat_model(is_flash=True)

def call_chatbot(state: State, config: RunnableConfig, store=None):
    """
    Standard chatbot node with 3-Tier Hierarchical Memory + Systemic Lessons.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info(f"Node: call_chatbot | Hierarchical Memory for User: {user_id}")
        last_user_msg = state["messages"][-1].content if state["messages"] else ""
        
        # --- LEVEL 1: IMMEDIATE CONTEXT (Short-Term) ---
        # Tier 1 & 2: Recent messages + Sliding window of history
        window_size = getattr(settings, "context_window_size", 20)
        current_chat_history = state["messages"][-window_size:]
        
        # --- LEVEL 2: KNOWLEDGE CONTEXT (Long-Term User Facts) ---
        memories = []
        if store is not None:
            # Search based on recent context
            memories = store.search((user_id, "memories"), query=last_user_msg, limit=5)
            if not memories:
                # Fallback to most recent facts if no semantic match
                memories = store.search((user_id, "memories"), query=None, limit=5)
        formatted_memories = "\n".join([f"- {m.value['fact']}" for m in memories]) if memories else "No user memories found."

        # --- LEVEL 3: SYSTEMIC CONTEXT (Lessons from Mistakes) ---
        lessons_text, applied_titles = get_relevant_lessons(last_user_msg, store)

        # --- ASSEMBLE & INVOKE ---
        prompt_template = get_assistant_prompt()
        chain = prompt_template | chat_model
        
        logger.info(f"Chatbot Node | Memory: {len(memories)} | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")
        
        response = chain.invoke({
            "memories": formatted_memories,
            "lessons": lessons_text,
            "tag": settings.memory_tag,
            "messages": current_chat_history
        })

        return {"messages": [response]}
    finally:
        log_context.reset(token)


def guardian_node(state: State, config: RunnableConfig, store=None):
    """Entry Point: Categorizes intent and enforces safety using Lessons."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info("Node: guardian_node")
        
        # 1. Global Rate Limit Check
        if not rate_limiter.check_and_record():
            logger.warning("Global Rate Limit Reached!")
            return {
                "intent": "blocked",
                "messages": [AIMessage(content="⚠️ System busy. Please wait a moment.")]
            }
        
        last_msg = state["messages"][-1].content
        lessons_text, applied_titles = get_relevant_lessons(last_msg, store)
        
        logger.info(f"Guardian Node | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")

        decision_prompt = f"""You are an Intent Classifier and Security Guardian.
Analyze the user message and categorize it into EXACTLY one of these intents:

1. SQL: Requesting database info (movies, actors, rentals, inventory).
2. BLOCKED: Requesting to DELETE, FORGET, or REMOVE personal facts or history from the user's memory.
3. CHAT: General conversation, greetings, personal preferences, or ASKING what the assistant knows about the user or anything isn't related to manipulation of memory is considered SAFE.
4. IRRELEVANT: For messages that are not about the Pagila DB, personal memory, or general chat assistance.

### RULES:
- Queries like "What do you know about me?", "Show my facts", or "What is my name?" are SAFE (Intent: CHAT).
- Queries like "Delete my facts", "Forget everything", or "Clear my history" are DANGEROUS (Intent: BLOCKED).

### PAST LESSONS:
{lessons_text}

User Message: "{last_msg}"
Rules:
- Output ONLY "INTENT: SQL", "INTENT: BLOCKED", "INTENT: CHAT", or "INTENT: IRRELEVANT".
"""
        res = guardian_model.invoke([SystemMessage(content=decision_prompt)]).content.strip().upper()
        
        if "BLOCKED" in res:
            logger.info("Guardian Action: BLOCKED")
            return {
                "intent": "blocked",
                "messages": [AIMessage(content="🛑 For your security, memory deletion can only be performed manually via the 'Long-Term Memory' manager in the sidebar.")]
            }
        elif "IRRELEVANT" in res:
            logger.info("Guardian Action: IRRELEVANT")
            return {
                "intent": "irrelevant",
                "messages": [AIMessage(content="I specialize in the Pagila DVD Rental database. How can I assist you with movie, actor, or rental data today?")]
            }
        elif "SQL" in res:
            logger.info("Guardian Action: SQL")
            return {"intent": "sql"}
        else:
            logger.info("Guardian Action: CHAT")
            return {"intent": "chat"}
    finally:
        log_context.reset(token)


def generate_sql_node(state: State, config: RunnableConfig, store=None):
    """Generates SQL query guided by relevant Lessons and Schema."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info("Node: generate_sql")
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        schema = sql_engine.get_schema()
        lessons_text, applied_titles = get_relevant_lessons(user_question, store)
        
        logger.info(f"SQL Generation Node | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")
        
        prompt_template = get_sql_generation_prompt()
        
        logger.debug(f"Generating SQL for question: {user_question}")
        
        chain = prompt_template | chat_model
        response = chain.invoke({
            "schema": schema,
            "lessons": lessons_text,
            "history": state["messages"][:-1],
            "question": user_question
        })
        
        sql_query = response.content.strip().replace("```sql", "").replace("```", "")
        logger.info(f"Generated SQL: {sql_query}")
        
        return {"current_sql": sql_query, "retry_count": 0}
    finally:
        log_context.reset(token)


def execute_sql_node(state: State, config: RunnableConfig):
    """
    Executes the current SQL query and captures results or errors.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info(f"Node: execute_sql | Query: {state['current_sql']}")
        result = sql_engine.execute_query(state["current_sql"])
        
        if result["status"] == "success":
            logger.info(f"Execution Success. Rows: {result['row_count']}")
            return {"sql_results": result["data"], "sql_error": None}
        else:
            logger.warning(f"Execution Failed. Error: {result['error_message']}")
            return {"sql_error": result["error_message"], "sql_results": []}
    finally:
        log_context.reset(token)


def heal_sql_node(state: State, config: RunnableConfig, store=None):
    """Heals SQL and records the lesson learned from the mistake."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        retry = state.get("retry_count", 0) + 1
        logger.info(f"Node: heal_sql | Attempt: {retry}")
        
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        schema = sql_engine.get_schema()
        
        prompt_template = get_sql_healing_prompt()

        logger.debug(f"Healing failed query. Error was: {state['sql_error']}")

        chain = prompt_template | chat_model
        response = chain.invoke({
            "schema": schema,
            "failed_query": state["current_sql"],
            "error_message": state["sql_error"],
            "question": user_question
        })
        
        fixed_sql = response.content.strip().replace("```sql", "").replace("```", "")
        
        # --- AUTOMATIC LEARNING LOOP ---
        if store:
            try:
                learning_prompt = f"""You are a Senior Staff Engineer. Analyze this SQL mistake and distilled fix.
Original Error: {state['sql_error']}
Fixed SQL: {fixed_sql}

Your task: Create a reusable Lesson.
1. Determine if this is a GLOBAL rule (applies to all queries) or SPECIFIC (only for this table/query).
2. Write a concise instruction for future models to avoid this mistake.

Output format:
TYPE: [GLOBAL/SPECIFIC]
TITLE: [Brief name]
INSTRUCTION: [The rule to follow]
REASONING: [Why we do this]
"""
                lesson_distillation = guardian_model.invoke([SystemMessage(content=learning_prompt)]).content
                
                is_global = "TYPE: GLOBAL" in lesson_distillation.upper()
                title = lesson_distillation.split("TITLE:")[1].split("\n")[0].strip()
                instruction = lesson_distillation.split("INSTRUCTION:")[1].split("\n")[0].strip()
                reasoning = lesson_distillation.split("REASONING:")[1].strip()
                
                record_lesson(title, state['sql_error'], instruction, reasoning, store, is_global=is_global)
            except Exception as e:
                logger.error(f"Failed to distill lesson: {e}")

        return {"current_sql": fixed_sql, "retry_count": retry}
    finally:
        log_context.reset(token)


def format_sql_response_node(state: State, config: RunnableConfig):
    """
    Python-based renderer that coordinates JSON output and Markdown table generation.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", settings.default_user_id)
    thread_id = configurable.get("thread_id", "unknown")
    
    token = log_context.set({"user_id": user_id, "thread_id": thread_id})
    try:
        logger.info("Node: format_sql_response")
        prompt_template = get_sql_response_format_prompt()
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

        # Request structured output from LLM
        chain = prompt_template | chat_model.with_structured_output(SQLResponse)
        
        try:
            response = chain.invoke({
                "question": user_question,
                "query": state["current_sql"],
                "data": str(state["sql_results"])
            })
            
            # --- Python Rendering Logic ---
            output_parts = []
            
            # 1. Table First (if exists)
            if response.table_data:
                table_md = generate_markdown_table(response.table_data)
                output_parts.append(table_md)
                
            # 2. Dedicated SQL Block
            if state.get("current_sql"):
                sql_block = f"**Executed SQL:**\n```sql\n{state['current_sql'].strip()}\n```"
                output_parts.append(sql_block)
                
            # 3. Summary Third (if exists)
            if response.summary:
                output_parts.append(response.summary)
                
            final_content = "\n\n".join(output_parts)
            
            if not final_content:
                final_content = "I found no results for your query in the database."
                
            return {"messages": [AIMessage(content=final_content)]}
            
        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            return {"messages": [AIMessage(content="I encountered an error while formatting the data.")]}
    finally:
        log_context.reset(token)
