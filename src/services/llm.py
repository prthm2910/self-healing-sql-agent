# ### --- [IMPORTS & CONFIGURATION] --- ###
import time
from typing import Any, Dict, Optional, Union, List

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from src.utils.limiter import rate_limiter
from src.core.config import settings
from src.utils.logger import logger


# ### --- [RATE-LIMITED CLIENT WRAPPER] --- ###

# [Elaborative Breakdown]
# API Resiliency and Token-Bucket Rate Limiting:
# Outbound calls to external LLM providers (like Groq) are highly vulnerable to
# rate limits (RPM/TPM - Requests/Tokens Per Minute) and sudden network failures.
# We wrap the core ChatGroq client in LoggedChatGroq to intercept all invokes,
# enforcing a token-bucket rate limiter immediately before sending requests.
#
# If a minute limit (RPM/TPM HTTP 429) is caught, the client executes an exponential
# wait-and-retry sequence. If a daily limit (RPD/TPD) is hit, it intercepts and throws
# a terminal exception immediately, preventing futile network calls and saving compute.
class LoggedChatGroq(ChatGroq):
    """A thread-safe, rate-limited, and self-healing subclass of ChatGroq.

    Provides automatic minute-limit backing off, token consumption recording, and
    immediate daily-quota exhaustion intercepting.
    """

    def invoke(
        self,
        input: Union[str, List[Any], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> BaseMessage:
        """Invokes the chat model with input, wrapping it in rate-limiting and retry logic.

        Args:
            input: The prompt string or list of structural chat messages.
            config: Optional system configurations, thread-ids, and callbacks.
            **kwargs: Extra parameters passed down directly to the underlying model.

        Returns:
            BaseMessage: The structured chat response message from the LLM model.

        Raises:
            RuntimeError: If rate limit timeout is reached, daily quota is fully exhausted,
                or if the model fails after maximum retry thresholds.
        """
        # Set conservative retry parameters to handle transient network hiccups
        max_retries: int = 3
        attempt: int = 0
        
        while attempt < max_retries:
            # 1. Acquire ticket from sliding-window Rate Limiter (blocks if minute RPM/TPM is saturated)
            if not rate_limiter.wait_and_record(timeout=30.0):
                logger.error("Global Rate Limit Timeout!")
                raise RuntimeError("API Rate Limit Exceeded and wait timeout reached.")

            logger.info(f"Invoking Groq ({self.model_name}) (Attempt {attempt + 1}/{max_retries})")
            
            try:
                # 2. Delegate directly to the parent ChatGroq client using standard LangChain protocol
                response = super().invoke(input, config, **kwargs)
                
                # 3. Dynamic Token Accounting: Extract actual tokens consumed from LLM response metadata
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    # Feed actual token count back to the sliding window tracker to maintain correct TPM
                    rate_limiter.record_usage(usage.get("total_tokens", 0))
                    logger.info(f"Groq invocation complete. Tokens: {usage}")
                return response
                
            except Exception as e:
                error_msg: str = str(e).lower()
                
                # 4. Error Diagnostics: Parse the error trace to detect HTTP 429 Rate Limit conditions
                if "429" in error_msg or "rate_limit" in error_msg:
                    # Daily Limit Check: Daily quotas are unrecoverable; terminate immediately to save cost.
                    is_daily: bool = "tpd" in error_msg or "rpd" in error_msg or "per day" in error_msg
                    
                    if is_daily:
                        logger.error("DAILY QUOTA EXCEEDED! Account exhausted for 24h.")
                        raise RuntimeError("Groq Daily Quota Exceeded. Please wait 24h or use a different key.") from e
                    
                    # Minute Limit Check: RPM/TPM limit hit; block calling thread for 10s and retry.
                    logger.warning("Minute Limit reached. Waiting 10s before retry...")
                    time.sleep(10.0)
                    attempt += 1
                    continue
                
                # Non-rate-limit errors are propagated up immediately for custom validators or healers to intercept
                logger.error(f"Groq invocation failed: {e}", exc_info=True)
                raise

        raise RuntimeError(f"Failed to invoke Groq after {max_retries} attempts.")


# ### --- [CLIENT FACTORY FUNCTION] --- ###

def get_llm() -> LoggedChatGroq:
    """Factory to instantiate and return the logged and rate-limited ChatGroq provider.

    Returns:
        LoggedChatGroq: The configured, ready-to-invoke LLM client instance.
    """
    model: str = settings.model_name
    api_key: str = settings.groq_api_key
    
    logger.debug(f"Instantiating ChatGroq (model: {model})")
    return LoggedChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0
    )
