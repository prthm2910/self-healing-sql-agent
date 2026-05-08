import time

from langchain_openai import ChatOpenAI

from src.utils.limiter import rate_limiter
from src.core.config import settings
from src.utils.logger import logger

class LoggedChatOpenAI(ChatOpenAI):
    """Wrapped ChatOpenAI provider with logging."""
    
    def invoke(self, input, config=None, **kwargs):
        
        # Ensure total API consumption stays within budget with a short retry loop
        max_retries = 3
        for attempt in range(max_retries):
            if rate_limiter.check_and_record():
                break
            
            if attempt < max_retries - 1:
                logger.warning(f"Rate limit reached. Retrying in 2s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(2)
            else:
                logger.error("Global Rate Limit Reached after retries!")
                raise RuntimeError("API Rate Limit Exceeded. Please try again in a minute.")

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
    logger.debug(f"Instantiating LLM (model: {model})")
    return LoggedChatOpenAI(
        model=model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nim_base_url,
        temperature=0
    )
