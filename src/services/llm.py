import time
import re
from typing import Any

from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.utils.limiter import rate_limiter
from src.utils.key_manager import groq_key_manager
from src.core.config import settings
from src.utils.logger import logger

class LoggedChatGroq(ChatGroq):
    """Wrapped ChatGroq with Multi-Key Rotation & Blacklisting."""
    
    def invoke(self, input, config=None, **kwargs):
        max_attempts = len(groq_key_manager.keys) or 1
        attempt = 0
        
        while attempt < max_attempts:
            # 1. Get a fresh key from the pool (Blocks if keys are in short-term cooldown)
            current_key = groq_key_manager.get_next_key_with_retry()
            if not current_key:
                logger.error("All Groq API keys are blacklisted (Daily Quota Exceeded)!")
                raise RuntimeError("All Groq API keys are blacklisted. Wait for 24h reset.")
            
            # 2. Update the key for this instance
            self.groq_api_key = SecretStr(current_key)
            
            # 3. Wait for rate limiter (Global)
            if not rate_limiter.wait_and_record(timeout=30.0):
                logger.error("Global Rate Limit Timeout!")
                raise RuntimeError("API Rate Limit Exceeded and wait timeout reached.")

            logger.info(f"Invoking Groq ({self.model_name}) with key {current_key[:8]}... (Attempt {attempt+1})")
            
            try:
                response = super().invoke(input, config, **kwargs)
                
                # Success - Log usage
                usage = getattr(response, 'usage_metadata', None)
                if usage:
                    logger.info(f"Groq invocation complete. Tokens: {usage}")
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Detect Rate Limit (HTTP 429)
                if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                    is_daily = "daily" in error_msg or "tokens per day" in error_msg or "requests per day" in error_msg
                    
                    duration = 86400 if is_daily else 10 # 24h for daily, 10s for minute
                    reason = "Daily Limit Reached (TPD/RPD)" if is_daily else "Minute Limit Reached (TPM/RPM)"
                    
                    # Blacklist this key
                    groq_key_manager.blacklist_key(current_key, duration_seconds=duration, reason=reason)
                    
                    if is_daily:
                        # For Daily limits, we switch keys and retry
                        attempt += 1
                        continue 
                    else:
                        # For Minute limits, we DO NOT switch keys. 
                        # We just retry the same key (the manager will block/wait).
                        # We don't increment attempt here to prevent failing a long task due to 10s pauses.
                        continue
                
                # For other errors, log and raise
                logger.error(f"Groq invocation failed: {e}", exc_info=True)
                raise

        raise RuntimeError(f"Failed to invoke Groq after {max_attempts} attempts due to rate limits.")

def get_llm():
    """Factory to get the requested logged LLM provider (Groq Only)."""
    model = settings.model_name
    
    logger.debug(f"Instantiating LoggedChatGroq (model: {model})")
    return LoggedChatGroq(
        model=model,
        temperature=0,
        max_tokens=2048
    )
