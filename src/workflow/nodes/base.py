import time
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from src.core.config import settings
from src.services.llm import get_llm
from src.utils.logger import log_context, logger
from src.workflow.state import State

llm = get_llm()


class BaseNode:
    """
    Base class for all workflow nodes to enforce DRY principles.
    Handles logging context (user_id, thread_id) and standard logging setup.
    """
    name: str = "base_node"

    def __call__(self, state: State, config: RunnableConfig = None, **kwargs) -> Dict[str, Any]:
        config = config or {}
        # Safely fetch the configuration parameters
        configurable = config.get("configurable", {}) if hasattr(config, "get") else {}
        user_id = configurable.get("user_id", settings.default_user_id)
        thread_id = configurable.get("thread_id", "unknown")
        
        token = log_context.set({"user_id": user_id, "thread_id": thread_id})
        try:
            logger.info(f"Node: {self.name}")
            return self.execute(state, config, user_id=user_id, thread_id=thread_id, **kwargs)
        finally:
            log_context.reset(token)

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def robust_invoke(chain, input, schema_class, max_retries=2):
        """
        Invokes a structured output chain with a fallback to manual JSON parsing if Groq's 
        'Tool choice is required' error occurs.
        """
        import json
        import re
        from langchain_core.runnables import RunnableSequence
        
        # 1. Try standard structured output
        try:
            return chain.invoke(input)
        except Exception as e:
            logger.warning(f"Structured output failed for {schema_class.__name__} (Attempt 1) with error: {e}. Attempting robust fallback...")

        # 2. Manual Fallback: Ask for raw JSON and parse it
        raw_llm = llm # Use the local llm reference
        
        # Resolve the prompt text from the chain and input
        prompt_text = ""
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
            
        fallback_prompt = f"""{prompt_text}

### OUTPUT INSTRUCTIONS:
You MUST output ONLY a valid JSON object matching this schema:
{schema_class.model_json_schema()}

Ensure the output is a single valid JSON block enclosed in ```json ... ```.
"""
        
        for attempt in range(max_retries):
            try:
                res = raw_llm.invoke(fallback_prompt)
                content = res.content
                
                # Extract JSON from code blocks
                json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                json_str = json_match.group(1) if json_match else content
                
                # Remove any trailing commas or markdown artifacts
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Parse and validate with Pydantic
                parsed = json.loads(json_str)
                return schema_class(**parsed)
            except Exception as retry_err:
                logger.error(f"Fallback attempt {attempt+1} failed for {schema_class.__name__}: {retry_err}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get valid structured output for {schema_class.__name__} after fallback attempts.") from retry_err
                time.sleep(1)
