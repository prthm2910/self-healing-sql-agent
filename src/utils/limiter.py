import time
import threading
from src.core.config import settings

class GlobalRateLimiter:
    """
    A thread-safe Global Rate Limiter to manage multi-agent API usage.
    Tracks both RPM (Requests Per Minute) and TPM (Tokens Per Minute).
    """
    def __init__(self, rpm_limit: int, tpm_limit: int = 500000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = [] # List of (timestamp)
        self.tokens = [] # List of (timestamp, token_count)
        self.lock = threading.Lock()

    def wait_and_record(self, timeout: float = 60.0) -> bool:
        """
        Blocks until both a request slot and token capacity are available.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                now = time.time()
                # Clean up old data (older than 60s)
                self.requests = [t for t in self.requests if now - t < 60]
                self.tokens = [item for item in self.tokens if now - item[0] < 60]
                
                # Check RPM
                if len(self.requests) >= self.rpm_limit:
                    time.sleep(0.5)
                    continue
                
                # Check TPM (Conservative estimate: assume new request takes 4k tokens if unknown)
                current_tpm = sum(item[1] for item in self.tokens)
                if current_tpm >= self.tpm_limit:
                    time.sleep(0.5)
                    continue
                
                # Capacity available! Record request
                self.requests.append(now)
                return True
            
        return False

    def record_usage(self, token_count: int):
        """Records actual token usage after a successful call."""
        with self.lock:
            self.tokens.append((time.time(), token_count))

    def check_and_record(self) -> bool:
        """Non-blocking check (Legacy support)."""
        return self.wait_and_record(timeout=0.1)

    def get_stats(self) -> dict:
        """Returns current usage stats."""
        with self.lock:
            now = time.time()
            self.requests = [t for t in self.requests if now - t < 60]
            self.tokens = [item for item in self.tokens if now - item[0] < 60]
            return {
                "rpm": len(self.requests),
                "rpm_limit": self.rpm_limit,
                "tpm": sum(item[1] for item in self.tokens),
                "tpm_limit": self.tpm_limit
            }

# Singleton Instance
# Groq Free Tier typically has 30 RPM and variable TPM.
rate_limiter = GlobalRateLimiter(rpm_limit=27, tpm_limit=500000)
