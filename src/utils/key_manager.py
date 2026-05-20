import time
import threading
from typing import List, Dict, Optional
from src.utils.logger import logger

class KeyPoolManager:
    """
    Manages a pool of API keys with rotation and intelligent blacklisting.
    Supports distinguishing between short-term (minute) and long-term (daily) limits.
    """
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.blacklist: Dict[str, float] = {} # key -> expiry_timestamp
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_key_with_retry(self, timeout: float = 60.0) -> Optional[str]:
        """
        Returns the current active key. 
        - If the key is under a short-term limit (TPM/RPM), it waits.
        - If the key is under a long-term limit (TPD/RPD), it switches to the next key permanently.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                now = time.time()
                # 1. Cleanup expired
                self.blacklist = {k: expiry for k, expiry in self.blacklist.items() if expiry > now}
                
                if not self.keys:
                    return None

                # 2. Check the current active key
                current_key = self.keys[self.current_index]
                
                if current_key not in self.blacklist:
                    return current_key
                
                # 3. Handle blacklisted current key
                expiry = self.blacklist[current_key]
                wait_time = expiry - now
                
                if wait_time > 300: # It's a Daily Limit (> 5 mins)
                    logger.warning(f"Key {self.current_index} (starts with {current_key[:8]}) hit Daily Limit. Switching to next key.")
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    
                    # Safety check: if we've looped back to a daily-blacklisted key, we are out of quota
                    next_key = self.keys[self.current_index]
                    if next_key in self.blacklist and (self.blacklist[next_key] - now) > 300:
                        logger.error("ALL Groq keys have hit their Daily Limit (TPD/RPD).")
                        return None
                    
                    continue # Immediate retry with new index
                
                # 4. It's a Minute Limit (TPM/RPM)
                # The user wants to wait for THIS key rather than switching
                logger.debug(f"Key {self.current_index} hit Minute Limit. Waiting {wait_time:.1f}s...")
            
            # Sleep and retry same key
            time.sleep(1.0)
            
        return None

    def blacklist_key(self, key: str, duration_seconds: int = 60, reason: str = "Rate limit"):
        """
        Blacklists a key for a specific duration.
        - Short duration (e.g., 60s) for TPM/RPM.
        - Long duration (e.g., 86400s / 24h) for TPD/RPD.
        """
        with self.lock:
            expiry = time.time() + duration_seconds
            self.blacklist[key] = expiry
            logger.warning(f"Key Blacklisted: {key[:8]}... for {duration_seconds}s. Reason: {reason}")

    def get_stats(self) -> dict:
        """Returns stats for the key pool."""
        with self.lock:
            now = time.time()
            active_count = sum(1 for k in self.keys if k not in self.blacklist or self.blacklist[k] <= now)
            return {
                "total_keys": len(self.keys),
                "active_keys": active_count,
                "blacklisted_keys": len(self.keys) - active_count
            }

# Initialize the manager with keys from settings
from src.core.config import settings
groq_key_manager = KeyPoolManager(settings.groq_keys)
