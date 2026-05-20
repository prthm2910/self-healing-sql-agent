import time
import threading
from src.core.config import settings

class GlobalRateLimiter:
    """
    A thread-safe Global Rate Limiter to manage multi-agent API usage.
    Tracks both RPM (Requests Per Minute) and TPM (Tokens Per Minute).
    """
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = []
        self.token_usage = [] # List of (timestamp, token_count)
        self.lock = threading.Lock()

    def _cleanup(self, now: float):
        """Removes entries older than 60 seconds."""
        self.requests = [t for t in self.requests if now - t < 60]
        self.token_usage = [item for item in self.token_usage if now - item[0] < 60]

    def wait_and_record(self, estimated_tokens: int = 500, timeout: float = 60.0) -> bool:
        """
        Blocks until both RPM and TPM capacity are available.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                now = time.time()
                self._cleanup(now)
                
                current_tpm = sum(item[1] for item in self.token_usage)
                
                # Check RPM and TPM safety (leave 5% buffer for TPM)
                if len(self.requests) < self.rpm_limit and (current_tpm + estimated_tokens) < (self.tpm_limit * 0.95):
                    self.requests.append(now)
                    # We don't record tokens yet; we wait for actual response metadata
                    return True
            
            # Capacity not available, sleep and retry
            time.sleep(1.0)
            
        return False

    def record_usage(self, actual_tokens: int):
        """Records actual token consumption from an LLM response."""
        with self.lock:
            self.token_usage.append((time.time(), actual_tokens))

    def check_and_record(self) -> bool:
        """Non-blocking check (Legacy support)."""
        with self.lock:
            now = time.time()
            self._cleanup(now)
            if len(self.requests) >= self.rpm_limit:
                return False
            self.requests.append(now)
            return True

    def get_stats(self) -> dict:
        """Returns current usage stats for UI transparency."""
        with self.lock:
            now = time.time()
            self._cleanup(now)
            return {
                "rpm": len(self.requests),
                "tpm": sum(item[1] for item in self.token_usage),
                "rpm_limit": self.rpm_limit,
                "tpm_limit": self.tpm_limit
            }

# Singleton Instance synchronized with global settings
rate_limiter = GlobalRateLimiter(
    rpm_limit=settings.rate_limit_rpm, 
    tpm_limit=settings.rate_limit_tpm
)
