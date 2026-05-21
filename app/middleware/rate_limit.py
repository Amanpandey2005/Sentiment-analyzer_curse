"""Redis-backed fixed-window rate limiting middleware."""

from __future__ import annotations

import time

import redis
from fastapi import HTTPException, Request, status

from app.core_config import get_settings


class RateLimiter:
    """Simple per-IP rate limiter with graceful local fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        self.local_counts: dict[str, tuple[int, int]] = {}

    def check(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        minute = int(time.time() // 60)
        limit = self.settings.api_rate_limit_per_minute
        key = f"rate:{client}:{minute}"
        try:
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, 70)
        except redis.RedisError:
            current_minute, count = self.local_counts.get(client, (minute, 0))
            count = count + 1 if current_minute == minute else 1
            self.local_counts[client] = (minute, count)

        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
