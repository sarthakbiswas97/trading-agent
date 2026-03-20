"""Redis event subscriber for inter-service communication."""

import json
import asyncio
import redis.asyncio as redis
from typing import Callable, Awaitable, Any
from datetime import datetime

from config import get_settings

settings = get_settings()


class EventSubscriber:
    """Subscribes to Redis pub/sub channels and streams."""

    def __init__(self):
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._handlers: dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._running: bool = False
        self._task: asyncio.Task | None = None

    async def connect(self):
        """Connect to Redis."""
        self._redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._redis.ping()
        self._pubsub = self._redis.pubsub()
        print("EventSubscriber connected to Redis")

    async def disconnect(self):
        """Disconnect from Redis."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None
        if self._redis:
            await self._redis.close()
            self._redis = None

    def on(self, channel: str, handler: Callable[[dict], Awaitable[None]]):
        """
        Register a handler for a channel.

        Args:
            channel: Redis pub/sub channel name
            handler: Async function to call with parsed message data
        """
        self._handlers[channel] = handler

    async def subscribe(self, *channels: str):
        """Subscribe to pub/sub channels."""
        if not self._pubsub:
            raise RuntimeError("Not connected to Redis")
        await self._pubsub.subscribe(*channels)
        print(f"Subscribed to channels: {channels}")

    async def start(self):
        """Start listening for messages."""
        if not self._pubsub:
            raise RuntimeError("Not connected to Redis")

        # Subscribe to all registered channels
        if self._handlers:
            await self.subscribe(*self._handlers.keys())

        self._running = True
        self._task = asyncio.create_task(self._listen())
        print("EventSubscriber started listening")

    async def _listen(self):
        """Internal message listening loop."""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message is None:
                    continue

                channel = message.get("channel")
                data = message.get("data")

                if channel and data and channel in self._handlers:
                    try:
                        parsed = json.loads(data)
                        await self._handlers[channel](parsed)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse message on {channel}: {e}")
                    except Exception as e:
                        print(f"Handler error on {channel}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Subscriber error: {e}")
                await asyncio.sleep(1)

    async def read_stream(
        self,
        stream: str,
        last_id: str = "$",
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, dict]]:
        """
        Read entries from a Redis stream.

        Args:
            stream: Stream name
            last_id: Start reading after this ID ("$" for new entries only)
            count: Max entries to return
            block_ms: Block timeout in milliseconds

        Returns:
            List of (entry_id, data) tuples
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        result = await self._redis.xread(
            {stream: last_id},
            count=count,
            block=block_ms,
        )

        if not result:
            return []

        entries = []
        for stream_name, messages in result:
            for entry_id, fields in messages:
                data = {k: self._deserialize(v) for k, v in fields.items()}
                entries.append((entry_id, data))

        return entries

    @staticmethod
    def _deserialize(value: str) -> Any:
        """Try to deserialize JSON strings."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value


# Global singleton
event_subscriber = EventSubscriber()
