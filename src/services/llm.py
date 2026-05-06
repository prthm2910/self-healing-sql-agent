from langchain_openai import ChatOpenAI
from src.core.config import settings
from src.utils.logger import logger

class LoggedChatOpenAI(ChatOpenAI):
    """Wrapped ChatOpenAI provider with logging."""
    
    def invoke(self, input, config=None, **kwargs):
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
