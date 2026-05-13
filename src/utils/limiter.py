import time
import threading
from src.core.config import settings

class GlobalRateLimiter:
    """
    A thread-safe Global Rate Limiter to manage multi-agent API usage.
    Ensures total requests stay under the given RPM cap.
    """
    def __init__(self, rpm_limit: int):
        self.rpm_limit = rpm_limit
        self.requests = []
        self.lock = threading.Lock()

    def wait_and_record(self, timeout: float = 60.0) -> bool:
        """
        Blocks until a request slot is available or timeout is reached.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                now = time.time()
                # Clean up old timestamps (older than 60s)
                self.requests = [t for t in self.requests if now - t < 60]
                
                if len(self.requests) < self.rpm_limit:
                    self.requests.append(now)
                    return True
            
            # Slot not available, wait a bit
            time.sleep(0.5)
            
        return False

    def check_and_record(self) -> bool:
        """Non-blocking check (Legacy support)."""
        with self.lock:
            now = time.time()
            self.requests = [t for t in self.requests if now - t < 60]
            if len(self.requests) >= self.rpm_limit:
                return False
            self.requests.append(now)
            return True

    def get_current_load(self) -> int:
        with self.lock:
            now = time.time()
            self.requests = [t for t in self.requests if now - t < 60]
            return len(self.requests)

# Singleton Instance for the entire App
rate_limiter = GlobalRateLimiter(rpm_limit=20)
