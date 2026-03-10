"""Market data service - fetches real-time and historical data from Binance."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Callable, Any
import aiohttp
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from config import get_settings
from db import get_session, Candle
from events.publisher import event_publisher

settings = get_settings()


@dataclass
class Tick:
    """Real-time price tick."""
    symbol: str
    price: float
    quantity: float
    timestamp: int  # Unix ms
    is_buyer_maker: bool


@dataclass
class CandleData:
    """OHLCV candle data."""
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    num_trades: int
    is_closed: bool


class MarketDataService:
    """Handles all market data operations."""

    BINANCE_REST_URL = "https://api.binance.com"
    BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str = "ETHUSDC"):
        self.symbol = symbol
        self.symbol_lower = symbol.lower()

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running = False

        # Callbacks for new data
        self._tick_callbacks: list[Callable[[Tick], Any]] = []
        self._candle_callbacks: list[Callable[[CandleData], Any]] = []

        # Latest data cache
        self.latest_price: float = 0.0
        self.latest_candle: CandleData | None = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    async def start(self):
        """Start the market data service."""
        print(f"Starting MarketDataService for {self.symbol}")

        # Connect to event publisher
        await event_publisher.connect()

        # Fetch historical data first
        await self._fetch_historical_candles(days=7)

        # Start WebSocket connection
        self._running = True
        asyncio.create_task(self._websocket_loop())

        print("MarketDataService started")

    async def stop(self):
        """Stop the market data service."""
        print("Stopping MarketDataService...")
        self._running = False

        if self._ws:
            await self._ws.close()
        if self._ws_session:
            await self._ws_session.close()

        await event_publisher.disconnect()
        print("MarketDataService stopped")

    def on_tick(self, callback: Callable[[Tick], Any]):
        """Register callback for new ticks."""
        self._tick_callbacks.append(callback)

    def on_candle(self, callback: Callable[[CandleData], Any]):
        """Register callback for new/updated candles."""
        self._candle_callbacks.append(callback)

    async def get_recent_candles(self, limit: int = 100) -> list[CandleData]:
        """Get recent candles from database."""
        async with get_session() as session:
            result = await session.execute(
                select(Candle)
                .where(Candle.symbol == self.symbol)
                .where(Candle.interval == "1m")
                .where(Candle.is_closed == True)
                .order_by(Candle.open_time.desc())
                .limit(limit)
            )
            rows = result.scalars().all()

            return [
                CandleData(
                    symbol=row.symbol,
                    interval=row.interval,
                    open_time=row.open_time,
                    close_time=row.close_time,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume,
                    quote_volume=row.quote_volume,
                    num_trades=row.num_trades,
                    is_closed=row.is_closed,
                )
                for row in reversed(rows)  # Return chronological order
            ]

    # ─────────────────────────────────────────────────────────────
    # HISTORICAL DATA
    # ─────────────────────────────────────────────────────────────

    async def _fetch_historical_candles(self, days: int = 7):
        """Fetch historical 1m candles from Binance REST API."""
        print(f"Fetching {days} days of historical candles...")

        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        all_candles = []
        current_start = start_time

        async with aiohttp.ClientSession() as session:
            while current_start < end_time:
                url = f"{self.BINANCE_REST_URL}/api/v3/klines"
                params = {
                    "symbol": self.symbol,
                    "interval": "1m",
                    "startTime": current_start,
                    "limit": 1000,  # Max per request
                }

                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        print(f"Error fetching candles: {resp.status}")
                        break

                    data = await resp.json()
                    if not data:
                        break

                    for kline in data:
                        candle = CandleData(
                            symbol=self.symbol,
                            interval="1m",
                            open_time=kline[0],
                            close_time=kline[6],
                            open=float(kline[1]),
                            high=float(kline[2]),
                            low=float(kline[3]),
                            close=float(kline[4]),
                            volume=float(kline[5]),
                            quote_volume=float(kline[7]),
                            num_trades=int(kline[8]),
                            is_closed=True,
                        )
                        all_candles.append(candle)

                    # Move to next batch
                    current_start = data[-1][6] + 1  # Last close_time + 1ms

                    # Rate limit: 1200 requests/min = 20/sec
                    await asyncio.sleep(0.1)

        # Store in database
        await self._store_candles(all_candles)
        print(f"Stored {len(all_candles)} historical candles")

    async def _store_candles(self, candles: list[CandleData]):
        """Store candles in PostgreSQL with upsert."""
        if not candles:
            return

        async with get_session() as session:
            for candle in candles:
                stmt = insert(Candle).values(
                    symbol=candle.symbol,
                    interval=candle.interval,
                    open_time=candle.open_time,
                    close_time=candle.close_time,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    quote_volume=candle.quote_volume,
                    num_trades=candle.num_trades,
                    is_closed=candle.is_closed,
                ).on_conflict_do_update(
                    index_elements=["symbol", "interval", "open_time"],
                    set_={
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                        "quote_volume": candle.quote_volume,
                        "num_trades": candle.num_trades,
                        "is_closed": candle.is_closed,
                    }
                )
                await session.execute(stmt)

    # ─────────────────────────────────────────────────────────────
    # WEBSOCKET STREAMING
    # ─────────────────────────────────────────────────────────────

    async def _websocket_loop(self):
        """Main WebSocket connection loop with reconnection logic."""
        reconnect_delay = 1

        while self._running:
            try:
                await self._connect_websocket()
                reconnect_delay = 1  # Reset on successful connection

            except Exception as e:
                print(f"WebSocket error: {e}")
                if self._running:
                    print(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff

    async def _connect_websocket(self):
        """Connect to Binance WebSocket and process messages."""
        streams = [
            f"{self.symbol_lower}@trade",      # Individual trades
            f"{self.symbol_lower}@kline_1m",   # 1-minute candles
        ]
        stream_path = "/".join(streams)
        ws_url = f"{self.BINANCE_WS_URL}/{stream_path}"

        print(f"Connecting to WebSocket: {ws_url}")

        self._ws_session = aiohttp.ClientSession()
        self._ws = await self._ws_session.ws_connect(ws_url)

        print("WebSocket connected")

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_ws_message(json.loads(msg.data))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {self._ws.exception()}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                print("WebSocket closed")
                break

    async def _handle_ws_message(self, data: dict):
        """Route WebSocket message to appropriate handler."""
        event_type = data.get("e")

        if event_type == "trade":
            await self._handle_trade(data)
        elif event_type == "kline":
            await self._handle_kline(data)

    async def _handle_trade(self, data: dict):
        """Handle individual trade message."""
        tick = Tick(
            symbol=data["s"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            timestamp=data["T"],
            is_buyer_maker=data["m"],
        )

        self.latest_price = tick.price

        # Store in Redis stream (last 1000 ticks)
        await event_publisher.add_to_stream(
            "stream:ticks",
            {
                "symbol": tick.symbol,
                "price": tick.price,
                "quantity": tick.quantity,
                "timestamp": tick.timestamp,
                "is_buyer_maker": tick.is_buyer_maker,
            },
            maxlen=1000,
        )

        # Update latest price in Redis
        await event_publisher.set_json(
            "market:latest_price",
            {"symbol": tick.symbol, "price": tick.price, "timestamp": tick.timestamp},
        )

        # Publish event
        await event_publisher.publish(
            "event:tick",
            {"symbol": tick.symbol, "price": tick.price, "timestamp": tick.timestamp},
        )

        # Call registered callbacks
        for callback in self._tick_callbacks:
            try:
                result = callback(tick)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Tick callback error: {e}")

    async def _handle_kline(self, data: dict):
        """Handle kline (candlestick) message."""
        k = data["k"]

        candle = CandleData(
            symbol=k["s"],
            interval=k["i"],
            open_time=k["t"],
            close_time=k["T"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            quote_volume=float(k["q"]),
            num_trades=k["n"],
            is_closed=k["x"],
        )

        self.latest_candle = candle

        # Store closed candles to database
        if candle.is_closed:
            await self._store_candles([candle])

            # Publish candle closed event (triggers feature computation)
            await event_publisher.publish(
                "event:candle_closed",
                {
                    "symbol": candle.symbol,
                    "interval": candle.interval,
                    "open_time": candle.open_time,
                    "close": candle.close,
                    "volume": candle.volume,
                },
            )

        # Update current candle in Redis
        await event_publisher.set_json(
            "market:current_candle",
            {
                "symbol": candle.symbol,
                "interval": candle.interval,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "is_closed": candle.is_closed,
            },
        )

        # Call registered callbacks
        for callback in self._candle_callbacks:
            try:
                result = callback(candle)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Candle callback error: {e}")


# Global singleton
market_data_service = MarketDataService()
