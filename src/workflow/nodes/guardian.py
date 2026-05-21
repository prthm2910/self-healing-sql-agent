from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.utils.logger import logger
from src.utils.limiter import rate_limiter
from src.workflow.schema.guardian import GuardianOutput, ClarificationOutput
from src.workflow.nodes.base import BaseNode, llm
from src.prompts.guardian import (
    get_locked_guardian_prompt,
    get_unlocked_guardian_prompt,
    get_clarification_prompt
)


class GuardianNode(BaseNode):
    """Entry Point: Categorizes intent as SQL, DENY, or CLARIFY. Supports Stateful Context Lock."""
    name = "guardian_node"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
        # 1. Global Rate Limit Check
        if not rate_limiter.check_and_record():
            logger.warning("Global Rate Limit Reached!")
            return {
                "intent": "DENY",
                "messages": [AIMessage(content="⚠️ System busy. Please wait a moment.")]
            }
        
        last_msg = state["messages"][-1].content
        is_locked = state.get("is_awaiting_clarification", False)
        vague_context = state.get("vague_query_context", "")

        domain_summary = """
        This database (Pagila) contains DVD rental business data:
        1. PEOPLE: Actors, Customers, Staff members.
        2. INVENTORY: Films, Categories, Languages, Inventories, Stores.
        3. BUSINESS: Rentals, Payments, Addresses, Cities, Countries.
        """

        if is_locked:
            prompt_template = get_locked_guardian_prompt()
            prompt_val = prompt_template.invoke({
                "vague_context": vague_context,
                "last_msg": last_msg,
                "domain_summary": domain_summary
            })
        else:
            prompt_template = get_unlocked_guardian_prompt()
            prompt_val = prompt_template.invoke({
                "last_msg": last_msg,
                "domain_summary": domain_summary
            })
        # Use structured output for determinism
        chain = llm.with_structured_output(GuardianOutput)
        res = self.robust_invoke(chain, prompt_val.to_messages(), GuardianOutput)
        res.node_name = "guardian"
        
        logger.info(f"Guardian Action: {res.intent} | Context Locked: {is_locked} | Thought: {res.thought_process}")

        # Update logs for observability
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())

        if res.intent == "DENY":
            # If we were locked but they pivot to something invalid, reset the lock
            return {
                "intent": "DENY",
                "is_awaiting_clarification": False,
                "vague_query_context": "",
                "agent_logs": logs,
                "messages": [AIMessage(content="I specialize exclusively in the Pagila DVD Rental database. I cannot assist with personal requests, general chat, or data modification.")]
            }
        
        if res.intent == "CLARIFY":
            return {
                "intent": "CLARIFY",
                "is_awaiting_clarification": True,
                "vague_query_context": last_msg if not is_locked else f"{vague_context} + {last_msg}",
                "agent_logs": logs
            }
        
        # If SQL intent is reached, reset the clarification lock
        return {
            "intent": "SQL", 
            "is_awaiting_clarification": False,
            "vague_query_context": "",
            "agent_logs": logs
        }


class ClarifyNode(BaseNode):
    """Asks the user for clarification when the intent is ambiguous."""
    name = "clarify_node"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        # Use structured output
        chain = llm.with_structured_output(ClarificationOutput)
        prompt_template = get_clarification_prompt()
        prompt_val = prompt_template.invoke({"last_msg": state['messages'][-1].content})
        res = self.robust_invoke(chain, prompt_val.to_messages(), ClarificationOutput)
        
        return {"messages": [AIMessage(content=res.clarification_question)]}


# Instantiate node callable objects
guardian_node = GuardianNode()
clarify_node = ClarifyNode()
