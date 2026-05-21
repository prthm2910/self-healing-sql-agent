# ### --- [IMPORTS] --- ###

import json
import re
import time
from typing import Dict, Any, Type, Optional, Union

from langchain_core.runnables import RunnableConfig, RunnableSequence
from pydantic import BaseModel

from src.core.config import settings
from src.services.llm import get_llm
from src.utils.logger import log_context, logger
from src.workflow.state import State

# ### --- [INITIALIZATION] --- ###

llm = get_llm()


# ### --- [BASE NODE INTERFACE] --- ###

class BaseNode:
    """
    Base class for all workflow nodes to enforce DRY principles.
    
    Orchestrates logging context injection (such as user_id and thread_id) 
    dynamically per-invocation, tracks node names, and standardizes standard
    execution wrappers.
    
    Attributes:
        name (str): Unique identifier for the workflow node, utilized in trace logs.
    """
    name: str = "base_node"

    def __call__(
        self, 
        state: State, 
        config: Optional[RunnableConfig] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Orchestrate the execution of the workflow node with dynamic context log tracking.

        Ensures logging metadata contains the correct user and thread information, 
        then delegates execution to the concrete implementation subclass.

        Args:
            state (State): The active workflow state containing all shared data pools.
            config (Optional[RunnableConfig]): The LangGraph runnable configuration instance.
            **kwargs (Any): Additional dynamic arguments passed to the executor.

        Returns:
            Dict[str, Any]: State updates to be merged back into the LangGraph state store.
        """
        config = config or {}
        # 1. Safely extract dynamic configuration parameters (user and thread details) from RunnableConfig
        configurable: Dict[str, Any] = config.get("configurable", {}) if hasattr(config, "get") else {}
        user_id: str = configurable.get("user_id", settings.default_user_id)
        thread_id: str = configurable.get("thread_id", "unknown")
        
        # 2. Bind logging context safely using thread-safe ContextVars to capture thread scope logs automatically
        token = log_context.set({"user_id": user_id, "thread_id": thread_id})
        try:
            logger.info(f"Executing Node: {self.name}")
            # 3. Delegate execution flow directly to subclass logic implementation
            return self.execute(state, config, user_id=user_id, thread_id=thread_id, **kwargs)
        finally:
            # 4. Lifecycle Cleanliness: Reset the ContextVar token inside 'finally' block to prevent cross-request leakage
            log_context.reset(token)

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute the concrete operational logic of the node.
        
        Subclasses must implement this method.

        Args:
            state (State): The active workflow state.
            config (RunnableConfig): LangGraph run configurations.
            user_id (str): The identified caller uuid.
            thread_id (str): The active chat session uuid.
            **kwargs (Any): Additional dynamic parameters.

        Returns:
            Dict[str, Any]: State modifications or updates.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    # ### --- [ROBUST PARSING UTILITIES] --- ###

    # [Elaborative Breakdown]
    # The `robust_invoke` method acts as a resilient validation safeguard designed to
    # defend against flaky structured JSON outputs from LLM APIs (e.g., Groq's tool usage errors).
    #
    # --- Mental Model & Fallback Flow ---
    # 1. Primary Attempt (Structured Invocation):
    #    The method first attempts to run the standard LangChain structured output chain. If the
    #    upstream API fulfills the tool-call request cleanly, execution succeeds immediately.
    # 2. Secondary Attempt (Defense-in-Depth Fallback):
    #    If the primary attempt raises any exception (e.g., API limits, tool-choice errors, invalid JSON), 
    #    we drop back to a strict prompt instruction strategy:
    #    - We reconstruct the prompt, explicitly injecting instructions to output JSON matching the
    #      Pydantic schema (obtained via `schema_class.model_json_schema()`).
    #    - We instruct the model to encapsulate the output in raw markdown JSON fences (```json ... ```).
    #    - We attempt a regular expression extract pattern (`r"```json\s*(.*?)\s*```"`) to capture the JSON substring,
    #      performing basic string scrubbing (e.g., stripping trailing commas in objects or lists) to ensure compatibility.
    #    - The resulting string is parsed with standard `json.loads` and instantiated back into the requested Pydantic 
    #      `schema_class`, raising a `RuntimeError` only after exhausting all retry budget.
    @staticmethod
    def robust_invoke(
        chain: Any, 
        input: Union[Dict[str, Any], List[Any], str], 
        schema_class: Type[BaseModel], 
        max_retries: int = 2
    ) -> Any:
        """
        Invoke a structured output chain with a resilient regex fallback to manual JSON parsing.

        Args:
            chain (Any): The primary LangChain structured output chain to invoke.
            input (Union[Dict[str, Any], List[Any], str]): Prompt arguments or raw message inputs.
            schema_class (Type[BaseModel]): The Pydantic V2 schema model class to validate and instantiate.
            max_retries (int): Number of fallback instruction generation iterations before failure.

        Returns:
            Any: An instance of `schema_class` containing verified structured data.

        Raises:
            RuntimeError: If all parsing and retry validation loops fail.
        """
        # 1. Primary Attempt: Try standard structured output
        try:
            return chain.invoke(input)
        except Exception as e:
            logger.warning(
                f"Structured output failed for {schema_class.__name__} (Attempt 1). Error: {e}. "
                f"Attempting robust instruction fallback..."
            )

        # 2. Secondary Attempt: Manual Fallback - Ask for raw JSON and parse it manually
        raw_llm = llm  # Retrieve the local rate-limited llm client
        
        # Resolve raw prompt text from the chain or inputs
        prompt_text: str = ""
        try:
            if isinstance(chain, RunnableSequence):
                prompt_val = chain.first.invoke(input)
                prompt_text = prompt_val.to_string() if hasattr(prompt_val, "to_string") else str(prompt_val)
            elif isinstance(input, list):
                prompt_text = "\n".join([m.content for m in input if hasattr(m, 'content')])
            else:
                prompt_text = str(input)
        except Exception as p_err:
            logger.error(f"Failed to resolve prompt text for fallback: {p_err}")
            prompt_text = str(input)
            
        # Reconstruct fallback instructions, injecting Pydantic json_schema
        fallback_prompt: str = f"""{prompt_text}

### OUTPUT INSTRUCTIONS:
You MUST output ONLY a valid JSON object matching this schema:
{schema_class.model_json_schema()}

Ensure the output is a single valid JSON block enclosed in ```json ... ```.
"""
        
        for attempt in range(max_retries):
            try:
                # Invoke core LLM client with plain instruction string
                res = raw_llm.invoke(fallback_prompt)
                content: str = res.content
                
                # Extract JSON substring from markdown fences (```json ... ```)
                json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                json_str: str = json_match.group(1) if json_match else content
                
                # Regex Scrubbing: Strip invalid trailing commas in JSON object lists/dicts (common LLM defect)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Parse validated string into dictionary
                parsed: Dict[str, Any] = json.loads(json_str)
                # Instantiate and return structured Pydantic object
                return schema_class(**parsed)
            except Exception as retry_err:
                logger.error(f"Fallback attempt {attempt + 1} failed for {schema_class.__name__}: {retry_err}")
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to get valid structured output for {schema_class.__name__} after fallback attempts."
                    ) from retry_err
                time.sleep(1)

