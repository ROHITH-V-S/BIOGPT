import time
import asyncio
from collections import defaultdict
from fastapi import Request, HTTPException, status
from app.config import settings

class SimpleRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.window = 60.0
        self.history = defaultdict(list)
        self._lock = asyncio.Lock()
        
    async def check(self, request: Request):
        if self.requests_per_minute <= 0:
            return
            
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        now = time.monotonic()
        async with self._lock:
            # Clean up old history
            self.history[client_ip] = [t for t in self.history[client_ip] if now - t < self.window]
            
            if len(self.history[client_ip]) >= self.requests_per_minute:
                # Find when the oldest request in the window expires
                retry_after = int(self.window - (now - self.history[client_ip][0]))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too Many Requests",
                    headers={"Retry-After": str(retry_after)},
                )
            self.history[client_ip].append(now)

rate_limiter = SimpleRateLimiter(settings.RATE_LIMIT_REQUESTS_PER_MINUTE)

async def rate_limit_dependency(request: Request):
    await rate_limiter.check(request)
