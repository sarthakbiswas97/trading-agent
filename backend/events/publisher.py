"""Redis event publisher for inter-service communication."""

import json
import redis.asyncio as redis
from typing import Any
from datetime import datetime

from config import get_settings

settings = get_settings()


class EventPublisher:
    """Publishes events to Redis pub/sub and streams."""

    def __init__(self):
        self._redis: redis.Redis | None = None

    async def connect(self):
        """Connect to Redis."""
        self._redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await self._redis.ping()
        print("EventPublisher connected to Redis")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def publish(self, channel: str, data: dict[str, Any]):
        """Publish event to pub/sub channel."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        message = json.dumps(data, default=self._serialize)
        await self._redis.publish(channel, message)

    async def add_to_stream(
        self,
        stream: str,
        data: dict[str, Any],
        maxlen: int = 1000,
    ):
        """Add entry to Redis stream with automatic trimming."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        # Flatten nested dicts for Redis stream
        flat_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                     for k, v in data.items()}

        await self._redis.xadd(stream, flat_data, maxlen=maxlen)

    async def get_stream_range(
        self,
        stream: str,
        count: int = 100,
        start: str = "-",
        end: str = "+",
    ) -> list[dict]:
        """Get entries from Redis stream."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        entries = await self._redis.xrange(stream, start, end, count=count)
        return [
            {"id": entry_id, **{k: self._deserialize(v) for k, v in fields.items()}}
            for entry_id, fields in entries
        ]

    async def get_latest_from_stream(self, stream: str, count: int = 1) -> list[dict]:
        """Get latest entries from stream."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        entries = await self._redis.xrevrange(stream, count=count)
        return [
            {"id": entry_id, **{k: self._deserialize(v) for k, v in fields.items()}}
            for entry_id, fields in entries
        ]

    async def set_json(self, key: str, data: dict[str, Any], expire_seconds: int | None = None):
        """Set JSON data with optional expiration."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        value = json.dumps(data, default=self._serialize)
        if expire_seconds:
            await self._redis.setex(key, expire_seconds, value)
        else:
            await self._redis.set(key, value)

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Get JSON data."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        value = await self._redis.get(key)
        if value:
            return json.loads(value)
        return None

    @staticmethod
    def _serialize(obj: Any) -> str:
        """Serialize non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    @staticmethod
    def _deserialize(value: str) -> Any:
        """Try to deserialize JSON strings."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value


# Global singleton
event_publisher = EventPublisher()
