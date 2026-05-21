# ### --- [IMPORTS] --- ###

from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.core.config import settings
from src.utils.table import generate_markdown_table
from src.services.lessons import get_relevant_lessons
from src.prompts.sql_agent import get_sql_response_format_prompt
from src.workflow.schema.response import ChatbotResponse, SQLResponse
from src.workflow.nodes.base import BaseNode, llm, logger


# ### --- [PROMPT UTILITIES] --- ###

def get_assistant_prompt(*args: Any, **kwargs: Any) -> Any:
    """
    Dynamically resolves assistant prompt based on module configuration mappings.

    Ensures that any custom assistant prompts override the default prompts.

    Args:
        *args (Any): Dynamic positional arguments.
        **kwargs (Any): Dynamic keyword arguments.

    Returns:
        Any: The resolved prompt template instance.
    """
    import sys
    nodes_mod = sys.modules.get("src.workflow.nodes")
    if nodes_mod and hasattr(nodes_mod, "get_assistant_prompt"):
        func = nodes_mod.get_assistant_prompt
        from src.prompts.assistant import get_assistant_prompt as real_func
        if func is not real_func:
            return func(*args, **kwargs)
    from src.prompts.assistant import get_assistant_prompt as real_func
    return real_func(*args, **kwargs)


# ### --- [CALL CHATBOT NODE] --- ###

class CallChatbotNode(BaseNode):
    """
    Standard chatbot node with Systemic Lessons.
    
    Acts as the primary conversational interaction node. It handles general questions
    by applying a sliding context window to recent conversation history and querying 
    our memory lesson store to dynamically adapt and avoid repeating past conversational 
    mistakes.
    """
    name: str = "call_chatbot"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        store: Optional[Any] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes assistant chatbot dialog operations using systemic memory lessons.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes with generated assistant chatbot response.
        """
        logger.info(f"Node: call_chatbot | User: {user_id}")
        last_user_msg: str = state["messages"][-1].content if state["messages"] else ""
        
        # 1. Sliding Window Context Truncation:
        # To maintain high performance and control costs, we enforce a strict sliding window 
        # on recent messages, feeding only the relevant conversational tail to the model.
        window_size: int = getattr(settings, "context_window_size", 20)
        current_chat_history: List[Any] = state["messages"][-window_size:]
        
        # 2. Semantic Memory Lookup:
        # Load relevant lessons dynamically from our vector-backed memory. This forces the LLM 
        # to adapt to past operational constraints or domain specific configurations.
        lessons_text, applied_titles = get_relevant_lessons(last_user_msg, store)

        # 3. Invoke Chatbot Prompt: Assemble prompt templates and run rate-limited structured LLM.
        prompt_template = get_assistant_prompt()
        chain = prompt_template | llm.with_structured_output(ChatbotResponse)
        
        logger.info(f"Chatbot Node | Lessons: {len(applied_titles)} {applied_titles if applied_titles else ''}")
        
        res: ChatbotResponse = self.robust_invoke(chain, {
            "lessons": lessons_text,
            "messages": current_chat_history
        }, ChatbotResponse)

        # Internal Validation -> Export to Dict
        res.node_name = "call_chatbot"
        return {"messages": [AIMessage(content=res.response)]}


# ### --- [FORMAT SQL RESPONSE NODE] --- ###

class FormatSQLResponseNode(BaseNode):
    """
    Ultra-low latency hybrid renderer node.
    
    Optimizes total execution times and token spends by executing a dual-rendering 
    strategy depending on query output structures:
    - Pure Python template + markdown visual grid formatting for non-aggregated records 
      (eliminating LLM summaries and costs completely).
    - LLM structured summaries only for aggregated queries where human context explanation 
      is valuable (e.g. summarizing total sales or actor counts).
    """
    name: str = "format_sql_response"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Renders database query results as user-facing markdown text blocks.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes containing rendered AIMessage response contents.
        """
        # 1. Fetch Execution States: Retrieve query string, raw result list, and the aggregated flag.
        user_question: str = next(
            m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        raw_results: List[Dict[str, Any]] = state.get("sql_results", [])
        is_aggregated: bool = state.get("is_aggregated", False)
        
        output_parts: List[str] = []
        
        # 2. Dual-Rendering Execution Strategy:
        if not raw_results:
            # Scenario A: Database returned zero rows.
            output_parts.append("No results found.")
        elif is_aggregated:
            # Scenario B: Aggregated single-cell result (e.g., SELECT count(*)).
            # Prompt the LLM to write a natural language summary to give context to this single number.
            prompt_template = get_sql_response_format_prompt()
            chain = prompt_template | llm.with_structured_output(SQLResponse)
            try:
                response: SQLResponse = self.robust_invoke(chain, {
                    "question": user_question,
                    "query": state.get("current_sql", ""),
                    "data": str(raw_results)
                }, SQLResponse)
                
                if response.summary:
                    output_parts.append(response.summary)
                else:
                    # Fallback if LLM output fails
                    val = list(raw_results[0].values())[0]
                    output_parts.append(f"The result is {val}.")
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                val = list(raw_results[0].values())[0]
                output_parts.append(f"The result is {val}.")
        else:
            # Scenario C (Zero-Latency Optimization): Detailed multi-row records.
            # Avoid sending long lists to the LLM to save token cost and eliminate API latency.
            # Instead, format them directly in Python using our fast GFM markdown table generator.
            row_count: int = len(raw_results)
            output_parts.append(f"Here are the details of the {row_count} results:")
            
            table_md: str = generate_markdown_table(raw_results)
            output_parts.append(table_md)
            
        # 3. Transparency Injection: Append the exact SQL statement executed to let the user review it.
        if state.get("current_sql"):
            sql_block: str = f"**Executed SQL:**\n```sql\n{state['current_sql'].strip()}\n```"
            output_parts.append(sql_block)
            
        # 4. Stitch output sections with double-newlines for clean presentation.
        final_content: str = "\n\n".join(output_parts)
        return {"messages": [AIMessage(content=final_content)]}


# ### --- [NODE INSTANTIATION] --- ###

# Instantiate node callable objects
call_chatbot = CallChatbotNode()
format_sql_response_node = FormatSQLResponseNode()


