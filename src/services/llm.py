import time

from langchain_openai import ChatOpenAI

from src.utils.limiter import rate_limiter
from src.core.config import settings
from src.utils.logger import logger

class LoggedChatOpenAI(ChatOpenAI):
    """Wrapped ChatOpenAI provider with logging."""
    
    def invoke(self, input, config=None, **kwargs):
        
        # Ensure total API consumption stays within budget by WAITING for a slot
        if not rate_limiter.wait_and_record(timeout=30.0):
            logger.error("Global Rate Limit Timeout!")
            raise RuntimeError("API Rate Limit Exceeded and wait timeout reached.")

        logger.info(f"Invoking LLM ({self.model_name or self.model})...")
        try:
            response = super().invoke(input, config, **kwargs)
            # Try to extract token usage if available
            usage = getattr(response, 'usage_metadata', None)
            if usage:
                logger.info(f"LLM invocation complete. Tokens: {usage}")
            else:
                logger.info("LLM invocation complete.")
            return response
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise

def get_llm():
    """Factory to get the requested logged LLM provider."""
    model = settings.model_name
    api_key = settings.nvidia_api_key
    base_url = settings.nim_base_url
    
    logger.debug(f"Instantiating LLM (model: {model}) at {base_url}")
    return LoggedChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0
    )
