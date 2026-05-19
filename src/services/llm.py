import time

from langchain_groq import ChatGroq

from src.utils.limiter import rate_limiter
from src.core.config import settings
from src.utils.logger import logger

class LoggedChatGroq(ChatGroq):
    """Wrapped ChatGroq provider with logging."""
    
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
    """Factory to get the requested logged LLM provider (Groq)."""
    model = settings.model_name
    api_key = settings.groq_api_key
    
    logger.debug(f"Instantiating ChatGroq (model: {model})")
    return LoggedChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0
    )
