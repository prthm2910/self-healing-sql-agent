from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.core.config import settings
from src.utils.table import generate_markdown_table
from src.services.lessons import get_relevant_lessons
from src.prompts.sql_agent import get_sql_response_format_prompt
from src.workflow.schema.response import ChatbotResponse, SQLResponse
from src.workflow.nodes.base import BaseNode, llm, logger

def get_assistant_prompt(*args, **kwargs):
    import sys
    nodes_mod = sys.modules.get("src.workflow.nodes")
    if nodes_mod and hasattr(nodes_mod, "get_assistant_prompt"):
        func = nodes_mod.get_assistant_prompt
        from src.prompts.assistant import get_assistant_prompt as real_func
        if func is not real_func:
            return func(*args, **kwargs)
    from src.prompts.assistant import get_assistant_prompt as real_func
    return real_func(*args, **kwargs)


class CallChatbotNode(BaseNode):
    """
    Standard chatbot node with Systemic Lessons.
    """
    name = "call_chatbot"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"Node: call_chatbot | User: {user_id}")
        last_user_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Recent messages + Sliding window of history
        window_size = getattr(settings, "context_window_size", 20)
        current_chat_history = state["messages"][-window_size:]
        
        # --- LEVEL 3: SYSTEMIC CONTEXT (Lessons from Mistakes) ---
        lessons_text, applied_titles = get_relevant_lessons(last_user_msg, store)

        # --- ASSEMBLE & INVOKE ---
        prompt_template = get_assistant_prompt()
        chain = prompt_template | llm.with_structured_output(ChatbotResponse)
        
        logger.info(f"Chatbot Node | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")
        
        res = self.robust_invoke(chain, {
            "lessons": lessons_text,
            "messages": current_chat_history
        }, ChatbotResponse)

        # Internal Validation -> Export to Dict
        res.node_name = "call_chatbot"
        return {"messages": [AIMessage(content=res.response)]}


class FormatSQLResponseNode(BaseNode):
    """
    Ultra-low latency hybrid renderer: 
    - Pure Python template + tables for non-aggregated queries (zero LLM overhead).
    - LLM-based summaries for aggregated queries.
    """
    name = "format_sql_response"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        user_question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        raw_results = state.get("sql_results", [])
        is_aggregated = state.get("is_aggregated", False)
        
        output_parts = []
        
        if not raw_results:
            output_parts.append("No results found.")
        elif is_aggregated:
            # For aggregated queries, use the LLM to write a natural summary
            prompt_template = get_sql_response_format_prompt()
            chain = prompt_template | llm.with_structured_output(SQLResponse)
            try:
                response = self.robust_invoke(chain, {
                    "question": user_question,
                    "query": state.get("current_sql", ""),
                    "data": str(raw_results)
                }, SQLResponse)
                
                if response.summary:
                    output_parts.append(response.summary)
                else:
                    val = list(raw_results[0].values())[0]
                    output_parts.append(f"The result is {val}.")
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                val = list(raw_results[0].values())[0]
                output_parts.append(f"The result is {val}.")
        else:
            # For list queries, use a standardized, clean template + markdown table (no LLM overhead!)
            row_count = len(raw_results)
            output_parts.append(f"Here are the details of the {row_count} results:")
            
            table_md = generate_markdown_table(raw_results)
            output_parts.append(table_md)
            
        # Append executed SQL code
        if state.get("current_sql"):
            sql_block = f"**Executed SQL:**\n```sql\n{state['current_sql'].strip()}\n```"
            output_parts.append(sql_block)
            
        final_content = "\n\n".join(output_parts)
        return {"messages": [AIMessage(content=final_content)]}


# Instantiate node callable objects
call_chatbot = CallChatbotNode()
format_sql_response_node = FormatSQLResponseNode()
