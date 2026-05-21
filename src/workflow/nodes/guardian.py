# ### --- [IMPORTS] --- ###

from typing import Dict, Any, List, Optional

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


# ### --- [GUARDIAN NODE] --- ###

class GuardianNode(BaseNode):
    """
    Entry Point: Categorizes intent as SQL, DENY, or CLARIFY. Supports Stateful Context Lock.
    
    This node serves as the primary gateway and security guard of our AI assistant.
    It evaluates every incoming message against global rate limits, checks for context
    locks (active clarification state), verifies that the query falls within the
    DVD rental (Pagila) domain boundaries, and determines correct downstream routing.
    """
    name: str = "guardian_node"

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
        Main routing gateway checking rate limits, security, and intent categories.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes with classified intent, logs, and optional deny messages.
        """
        # 1. Global Rate Limit Check:
        # Before sending queries to the external LLM provider, we intercept them and verify if
        # our sliding-window capacity permits another call, defending our API quotas from spikes.
        if not rate_limiter.check_and_record():
            logger.warning("Global Rate Limit Reached!")
            return {
                "intent": "DENY",
                "messages": [AIMessage(content="⚠️ System busy. Please wait a moment.")]
            }
        
        # 2. Context Lock Evaluation:
        # We retrieve the latest message. If the user was in an active clarification conversation block,
        # we load the vague context variable to guide the security classification.
        last_msg: str = state["messages"][-1].content
        is_locked: bool = state.get("is_awaiting_clarification", False)
        vague_context: str = state.get("vague_query_context", "")

        domain_summary: str = """
        This database (Pagila) contains DVD rental business data:
        1. PEOPLE: Actors, Customers, Staff members.
        2. INVENTORY: Films, Categories, Languages, Inventories, Stores.
        3. BUSINESS: Rentals, Payments, Addresses, Cities, Countries.
        """

        # 3. Dynamic Prompt Compilation:
        # Load appropriate guardian instructions depending on if the user is in a context lock.
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
            
        # 4. Invoke LLM Gateway Classifier: Run structured classification query.
        chain = llm.with_structured_output(GuardianOutput)
        res: GuardianOutput = self.robust_invoke(chain, prompt_val.to_messages(), GuardianOutput)
        res.node_name = "guardian"
        
        logger.info(f"Guardian Action: {res.intent} | Context Locked: {is_locked} | Thought: {res.thought_process}")

        # 5. Observability logging
        logs: List[Dict[str, Any]] = state.get("agent_logs", [])
        logs.append(res.model_dump())

        # 6. Intent Routing and Context Lock State Changes:
        if res.intent == "DENY":
            # Scenario A (Security Guard Triggered): User query violates DVD domain boundaries.
            # Reset active locks and return domain guard message.
            return {
                "intent": "DENY",
                "is_awaiting_clarification": False,
                "vague_query_context": "",
                "agent_logs": logs,
                "messages": [AIMessage(content="I specialize exclusively in the Pagila DVD Rental database. I cannot assist with personal requests, general chat, or data modification.")]
            }
        
        if res.intent == "CLARIFY":
            # Scenario B (Clarification Gating): Query is vague. Lock conversation state and request info.
            return {
                "intent": "CLARIFY",
                "is_awaiting_clarification": True,
                "vague_query_context": last_msg if not is_locked else f"{vague_context} + {last_msg}",
                "agent_logs": logs
            }
        
        # Scenario C (Permitted Query): Intent classified as SQL. Clear any legacy clarification locks and proceed.
        return {
            "intent": "SQL", 
            "is_awaiting_clarification": False,
            "vague_query_context": "",
            "agent_logs": logs
        }


# ### --- [CLARIFY NODE] --- ###

class ClarifyNode(BaseNode):
    """
    Asks the user for clarification when the intent is ambiguous.
    
    Generates friendly user-facing questions prompting them to provide missing context
    required to successfully target tables and generate executable SQL queries.
    """
    name: str = "clarify_node"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Constructs and runs the clarification message prompt interface.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes with generated clarification message.
        """
        # 1. Ask for Clarification: Run structured output chain to generate targeted question.
        chain = llm.with_structured_output(ClarificationOutput)
        prompt_template = get_clarification_prompt()
        prompt_val = prompt_template.invoke({"last_msg": state['messages'][-1].content})
        res: ClarificationOutput = self.robust_invoke(chain, prompt_val.to_messages(), ClarificationOutput)
        
        # 2. Return user-facing question inside LangGraph messages list
        return {"messages": [AIMessage(content=res.clarification_question)]}


# ### --- [NODE INSTANTIATION] --- ###

# Instantiate node callable objects
guardian_node = GuardianNode()
clarify_node = ClarifyNode()


