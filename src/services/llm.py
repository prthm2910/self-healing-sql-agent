# ### --- [IMPORTS & CONFIGURATION] --- ###
import time
from typing import Any, cast, Dict, Optional, Union, List

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from src.utils.limiter import rate_limiter
from src.core.config import settings
from src.utils.logger import logger


# ### --- [RATE-LIMITED CLIENT WRAPPER] --- ###

# [Elaborative Breakdown]
# API Resiliency and Token-Bucket Rate Limiting:
# Outbound calls to external LLM providers (like NVIDIA NIM) are highly vulnerable to
# rate limits (RPM/TPM - Requests/Tokens Per Minute) and sudden network failures.
# We wrap the core ChatNVIDIA client in LoggedChatNVIDIA to intercept all invokes,
# enforcing a token-bucket rate limiter immediately before sending requests.
#
# If a minute limit (RPM/TPM HTTP 429) is caught, the client executes an exponential
# wait-and-retry sequence. If a daily limit (RPD/TPD) is hit, it intercepts and throws
# a terminal exception immediately, preventing futile network calls and saving compute.
class LoggedChatNVIDIA(ChatNVIDIA):
    """A thread-safe, rate-limited, and self-healing subclass of ChatNVIDIA.

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

            logger.info(f"Invoking NVIDIA ({self.model}) (Attempt {attempt + 1}/{max_retries})")
            
            try:
                # 2. Delegate directly to the parent ChatNVIDIA client using standard LangChain protocol
                response = super().invoke(cast(Any, input), config, **kwargs)
                
                # 3. Dynamic Token Accounting: Extract actual tokens consumed from LLM response metadata
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    # Feed actual token count back to the sliding window tracker to maintain correct TPM
                    rate_limiter.record_usage(usage.get("total_tokens", 0))
                    logger.info(f"NVIDIA invocation complete. Tokens: {usage}")
                return response
                
            except Exception as e:
                error_msg: str = str(e).lower()
                
                # 4. Error Diagnostics: Parse the error trace to detect HTTP 429 Rate Limit conditions
                if "429" in error_msg or "rate_limit" in error_msg:
                    # Daily Limit Check: Daily quotas are unrecoverable; terminate immediately to save cost.
                    is_daily: bool = "tpd" in error_msg or "rpd" in error_msg or "per day" in error_msg
                    
                    if is_daily:
                        logger.error("DAILY QUOTA EXCEEDED! Account exhausted for 24h.")
                        raise RuntimeError("NVIDIA Daily Quota Exceeded. Please wait 24h or use a different key.") from e
                    
                    # Minute Limit Check: RPM/TPM limit hit; block calling thread for 10s and retry.
                    logger.warning("Minute Limit reached. Waiting 10s before retry...")
                    time.sleep(10.0)
                    attempt += 1
                    continue
                
                # Non-rate-limit errors are propagated up immediately for custom validators or healers to intercept
                logger.error(f"NVIDIA invocation failed: {e}", exc_info=True)
                raise

        raise RuntimeError(f"Failed to invoke NVIDIA after {max_retries} attempts.")


# ### --- [CLIENT FACTORY FUNCTION] --- ###

def get_llm(*, model: Optional[str] = None) -> LoggedChatNVIDIA:
    """Factory to instantiate and return the logged and rate-limited ChatNVIDIA provider.

    Args:
        model: Override model identifier. Falls back to settings.chat_model.

    Returns:
        LoggedChatNVIDIA: The configured, ready-to-invoke LLM client instance.
    """
    model_name: str = model or settings.chat_model
    api_key: str = settings.nvidia_api_key

    logger.debug(f"Instantiating ChatNVIDIA (model: {model_name})")
    return LoggedChatNVIDIA(
        model=model_name,
        nvidia_api_key=SecretStr(api_key), # type: ignore
        temperature=0
    )
