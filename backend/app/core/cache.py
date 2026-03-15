import time
from typing import Dict, Any, Optional

class DashboardCache:
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: Any):
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

    def clear(self):
        self.cache = {}

# Global instance for the application
dashboard_cache = DashboardCache(ttl_seconds=300)  # 5 minutes cache
