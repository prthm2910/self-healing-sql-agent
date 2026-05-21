import time

from langchain_groq import ChatGroq

from src.utils.limiter import rate_limiter
from src.core.config import settings
from src.utils.logger import logger


class LoggedChatGroq(ChatGroq):
    """Wrapped ChatGroq with integrated rate limiting and retry logic."""
    
    def invoke(self, input, config=None, **kwargs):
        max_retries = 3
        attempt = 0
        
        while attempt < max_retries:
            # 1. Wait for Global Capacity (Minute Limit)
            if not rate_limiter.wait_and_record(timeout=30.0):
                logger.error("Global Rate Limit Timeout!")
                raise RuntimeError("API Rate Limit Exceeded and wait timeout reached.")

            logger.info(f"Invoking Groq ({self.model_name}) (Attempt {attempt+1})")
            
            try:
                response = super().invoke(input, config, **kwargs)
                
                # Record usage on success
                usage = getattr(response, 'usage_metadata', None)
                if usage:
                    rate_limiter.record_usage(usage.get("total_tokens", 0))
                    logger.info(f"Groq invocation complete. Tokens: {usage}")
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for Rate Limit (HTTP 429)
                if "429" in error_msg or "rate_limit" in error_msg:
                    # Detect if it's a Daily Limit (Unrecoverable today)
                    is_daily = "tpd" in error_msg or "rpd" in error_msg or "per day" in error_msg
                    
                    if is_daily:
                        logger.error("DAILY QUOTA EXCEEDED! Account exhausted for 24h.")
                        raise RuntimeError("Groq Daily Quota Exceeded. Please wait 24h or use a different key.") from e
                    
                    # For Minute limits (RPM/TPM), wait 10s and retry
                    logger.warning("Minute Limit reached. Waiting 10s before retry...")
                    time.sleep(10.0)
                    attempt += 1
                    continue
                
                # For 400 errors (Tool failure), let robust_invoke handle it
                # For other errors, raise
                logger.error(f"Groq invocation failed: {e}", exc_info=True)
                raise

        raise RuntimeError(f"Failed to invoke Groq after {max_retries} attempts.")

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
