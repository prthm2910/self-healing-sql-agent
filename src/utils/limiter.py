# ### --- [IMPORTS] --- ###

import time
import threading
from typing import Dict, Any, List, Tuple

from src.core.config import settings

# ### --- [LIMIT CONSTANTS] --- ###

RPM_LIMIT: int = settings.rate_limit_rpm
TPM_LIMIT: int = settings.token_per_minute


# ### --- [GLOBAL RATE LIMITER] --- ###

# [Elaborative Breakdown]
# The `GlobalRateLimiter` class implements a thread-safe sliding-window log rate limiter.
#
# --- Why Slide-Window Throttling is Superior to Fixed Windows ---
# Fixed-window throttling resets counters at specific clock intervals (e.g., every hour). This creates 
# a vulnerability where double the allowed quota can be consumed within a brief period (e.g., at the 
# boundary edge of a window reset). 
# Sliding-window log rate limiting, however, continuously sweeps and purges any requests or token 
# records that are older than the specified time delta (60 seconds). This enforces a constant, 
# smooth, and rigid quota cap at any arbitrary sub-minute window.
#
# --- Concurrency & Thread-Safety Mechanism ---
# To support concurrent orchestration (such as Map-Reduce parallel worker nodes), this limiter uses 
# a reentrant thread `threading.Lock()`.
# - All state manipulation operations (appending request timestamps, sweeping expired entries, or calculating 
#   active token aggregates) are executed inside critical sections protected by `with self.lock:`.
# - If limit thresholds are reached, thread execution yields control to the CPU (`time.sleep(0.5)`) to permit 
#   competing threads to complete their operations, implementing a polling-backoff retry pattern.
class GlobalRateLimiter:
    """
    A thread-safe Global Rate Limiter to manage multi-agent API usage.
    
    Tracks both RPM (Requests Per Minute) and TPM (Tokens Per Minute) dynamically 
    using a sliding window history queue.
    
    Attributes:
        rpm_limit (int): Maximum number of requests allowed per 60-second window.
        tpm_limit (int): Maximum number of LLM tokens allowed per 60-second window.
        requests (List[float]): Active queue of timestamp markers for requests.
        tokens (List[Tuple[float, int]]): Active queue of timestamp and token weight pairings.
        lock (threading.Lock): Concurrency lock safeguarding queue mutations.
    """

    def __init__(self, rpm_limit: int = RPM_LIMIT, tpm_limit: int = TPM_LIMIT) -> None:
        """
        Initialize the global rate limiter with specified limits.

        Args:
            rpm_limit (int): Request rate limit per minute.
            tpm_limit (int): Token rate limit per minute.
        """
        self.rpm_limit: int = rpm_limit
        self.tpm_limit: int = tpm_limit
        self.requests: List[float] = []
        self.tokens: List[Tuple[float, int]] = []
        self.lock: threading.Lock = threading.Lock()

    def wait_and_record(self, timeout: float = 60.0) -> bool:
        """
        Blocks until both a request slot and token capacity are available or timeout is reached.

        Continuously sweeps expired queue items (older than 60 seconds) and checks active
        utilization before registering a new request ticket.

        Args:
            timeout (float): Max blocking duration in seconds before returning failure.

        Returns:
            bool: True if slot was successfully allocated, False if timeout elapsed.
        """
        start_time: float = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                now: float = time.time()
                # 1. Sliding Window Purge (Request & Token Cleanup):
                # We continuously sweep and discard any log entries that are older than 60 seconds.
                # This ensures our active capacity aggregates strictly represent the rolling sub-minute window.
                self.requests = [t for t in self.requests if now - t < 60]
                self.tokens = [item for item in self.tokens if now - item[0] < 60]
                
                # 2. Requests Per Minute (RPM) Gating Check:
                # If the number of requests registered in the active window meets or exceeds our RPM limit,
                # we block execution, sleep for 500ms, and retry.
                if len(self.requests) >= self.rpm_limit:
                    time.sleep(0.5)
                    continue
                
                # 3. Tokens Per Minute (TPM) Gating Check:
                # We calculate rolling token usage. Because we cannot know exact response tokens before calling
                # the API, we use a conservative approach: if the rolling TPM aggregate exceeds limits, 
                # we block the thread to avoid HTTP 429 errors from the LLM provider.
                current_tpm: int = sum(item[1] for item in self.tokens)
                if current_tpm >= self.tpm_limit:
                    time.sleep(0.5)
                    continue
                
                # 4. Capacity Available: Register active request timestamp in queue.
                self.requests.append(now)
                return True
            
        return False

    def record_usage(self, token_count: int) -> None:
        """
        Record actual token consumption after a successful API call.

        Args:
            token_count (int): Amount of tokens returned from LLM metrics metadata.
        """
        # Lock and append actual token count and timestamp to tokens sliding window.
        with self.lock:
            self.tokens.append((time.time(), token_count))

    def check_and_record(self) -> bool:
        """
        Execute a quick, non-blocking rate check with immediate feedback.

        Returns:
            bool: True if rate limits are currently cleared and recorded, False otherwise.
        """
        # Runs wait_and_record with a tiny timeout (100ms) for fast non-blocking triage.
        return self.wait_and_record(timeout=0.1)

    def get_stats(self) -> Dict[str, int]:
        """
        Retrieve a snapshot of active resource limits and sliding utilization.

        Returns:
            Dict[str, int]: Dict containing metrics for active rpm, limit, tpm, and tpm limit.
        """
        with self.lock:
            now: float = time.time()
            # Clean up old data to ensure metrics returned are fresh
            self.requests = [t for t in self.requests if now - t < 60]
            self.tokens = [item for item in self.tokens if now - item[0] < 60]
            return {
                "rpm": len(self.requests),
                "rpm_limit": self.rpm_limit,
                "tpm": sum(item[1] for item in self.tokens),
                "tpm_limit": self.tpm_limit
            }


# ### --- [SINGLETON INSTANTIATION] --- ###

rate_limiter = GlobalRateLimiter(rpm_limit=RPM_LIMIT, tpm_limit=TPM_LIMIT)

