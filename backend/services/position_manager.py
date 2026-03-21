"""
Position Manager - Tracks current trading position in Redis.

Manages:
- Opening/closing positions
- Tracking unrealized PnL
- Position state persistence
- Capital tracking and persistence
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field, asdict

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from events.publisher import event_publisher
from db.database import get_session
from db.models import CapitalSnapshot


@dataclass
class PositionState:
    """Current position state."""
    id: str = ""
    asset: str = "ETH"
    side: str = ""  # "LONG" or empty
    size: float = 0.0  # Amount of asset
    entry_price: float = 0.0
    current_price: float = 0.0
    entry_time: str = ""  # ISO timestamp
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.size > 0 and self.side != ""

    @property
    def value_usd(self) -> float:
        return self.size * self.current_price

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PositionState":
        return cls(**data)

    @classmethod
    def empty(cls) -> "PositionState":
        return cls()


class PositionManager:
    """
    Manages position state in Redis.

    Uses Redis for hot state with PostgreSQL for audit trail.
    """

    POSITION_KEY = "position:current"
    POSITION_HISTORY_STREAM = "stream:positions"

    def __init__(self, initial_capital: float = 10000.0):
        self.capital = initial_capital
        self._position: PositionState = PositionState.empty()

    @property
    def position(self) -> PositionState:
        return self._position

    @property
    def has_position(self) -> bool:
        return self._position.is_open

    async def load_from_redis(self):
        """Load position state from Redis."""
        data = await event_publisher.get_json(self.POSITION_KEY)
        if data:
            self._position = PositionState.from_dict(data)
        else:
            self._position = PositionState.empty()

    async def save_to_redis(self):
        """Save position state to Redis."""
        await event_publisher.set_json(
            self.POSITION_KEY,
            self._position.to_dict(),
        )

    async def open_position(
        self,
        side: str,
        size: float,
        entry_price: float,
        decision_id: str,
    ) -> PositionState:
        """
        Open a new position.

        Args:
            side: "LONG" (we only support long for now)
            size: Amount of asset to buy
            entry_price: Entry price in USDC
            decision_id: ID of decision that triggered this

        Returns:
            New position state
        """
        if self.has_position:
            raise ValueError("Already have an open position")

        now = datetime.now(timezone.utc)

        self._position = PositionState(
            id=str(uuid.uuid4()),
            asset="ETH",
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=now.isoformat(),
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
        )

        await self.save_to_redis()

        # Log to stream
        await event_publisher.add_to_stream(
            self.POSITION_HISTORY_STREAM,
            {
                "event": "OPEN",
                "position_id": self._position.id,
                "decision_id": decision_id,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "timestamp": now.isoformat(),
            },
            maxlen=1000,
        )

        print(f"Opened {side} position: {size} {self._position.asset} @ ${entry_price:.2f}")
        return self._position

    async def close_position(
        self,
        exit_price: float,
        reason: str,
        decision_id: str,
    ) -> tuple[PositionState, float]:
        """
        Close current position.

        Args:
            exit_price: Exit price in USDC
            reason: Reason for closing (stop_loss, take_profit, reversal, manual)
            decision_id: ID of decision that triggered this

        Returns:
            Tuple of (closed position state, realized PnL)
        """
        if not self.has_position:
            raise ValueError("No position to close")

        # Calculate realized PnL
        realized_pnl = (exit_price - self._position.entry_price) * self._position.size
        realized_pnl_pct = (exit_price - self._position.entry_price) / self._position.entry_price

        closed_position = PositionState(
            id=self._position.id,
            asset=self._position.asset,
            side=self._position.side,
            size=self._position.size,
            entry_price=self._position.entry_price,
            current_price=exit_price,
            entry_time=self._position.entry_time,
            unrealized_pnl=realized_pnl,
            unrealized_pnl_pct=realized_pnl_pct,
        )

        now = datetime.now(timezone.utc)

        # Log to stream
        await event_publisher.add_to_stream(
            self.POSITION_HISTORY_STREAM,
            {
                "event": "CLOSE",
                "position_id": self._position.id,
                "decision_id": decision_id,
                "reason": reason,
                "entry_price": self._position.entry_price,
                "exit_price": exit_price,
                "size": self._position.size,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
                "timestamp": now.isoformat(),
            },
            maxlen=1000,
        )

        print(f"Closed position @ ${exit_price:.2f} | PnL: ${realized_pnl:.2f} ({realized_pnl_pct:+.2%}) | Reason: {reason}")

        # Reset position
        self._position = PositionState.empty()
        await self.save_to_redis()

        return closed_position, realized_pnl

    async def update_price(self, current_price: float):
        """
        Update current price and recalculate unrealized PnL.

        Args:
            current_price: Current market price
        """
        if not self.has_position:
            return

        self._position.current_price = current_price
        self._position.unrealized_pnl = (
            (current_price - self._position.entry_price) * self._position.size
        )
        self._position.unrealized_pnl_pct = (
            (current_price - self._position.entry_price) / self._position.entry_price
        )

        await self.save_to_redis()

    def calculate_position_size(
        self,
        price: float,
        position_pct: float = 0.03,
    ) -> float:
        """
        Calculate position size in asset units.

        Args:
            price: Current price
            position_pct: Percentage of capital to use (default 3%)

        Returns:
            Size in asset units (e.g., ETH)
        """
        value_usd = self.capital * position_pct
        size = value_usd / price
        return round(size, 6)

    def get_position_age_seconds(self) -> float:
        """Get how long the current position has been open."""
        if not self.has_position or not self._position.entry_time:
            return 0.0

        entry = datetime.fromisoformat(self._position.entry_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - entry).total_seconds()

    # ================================================================
    # CAPITAL TRACKING & PERSISTENCE
    # ================================================================

    def get_current_capital(self) -> float:
        """
        Get current capital including unrealized PnL.

        Returns:
            Current portfolio value in USD
        """
        if self.has_position:
            return self.capital + self._position.unrealized_pnl
        return self.capital

    def update_capital(self, realized_pnl: float):
        """
        Update capital after closing a position.

        Args:
            realized_pnl: Realized profit/loss in USD
        """
        self.capital += realized_pnl

    async def save_capital_to_db(
        self,
        peak_capital: float,
        current_drawdown_pct: float,
        max_drawdown_pct: float,
        session_id: str = None,
    ):
        """
        Save capital state to PostgreSQL for persistence.

        Called on graceful shutdown or periodically for recovery.

        Args:
            peak_capital: Peak capital (high water mark)
            current_drawdown_pct: Current drawdown percentage
            max_drawdown_pct: Maximum drawdown ever seen
            session_id: Optional session identifier
        """
        try:
            snapshot_id = str(uuid.uuid4())
            now = datetime.utcnow()

            async with get_session() as session:
                # Mark all previous snapshots as not latest
                await session.execute(
                    update(CapitalSnapshot)
                    .where(CapitalSnapshot.is_latest == True)
                    .values(is_latest=False)
                )

                # Insert new snapshot as latest
                snapshot = CapitalSnapshot(
                    id=snapshot_id,
                    timestamp=now,
                    capital=self.capital,
                    peak_capital=peak_capital,
                    current_drawdown_pct=current_drawdown_pct,
                    max_drawdown_pct=max_drawdown_pct,
                    session_id=session_id,
                    is_latest=True,
                )
                session.add(snapshot)

            print(f"Capital snapshot saved: ${self.capital:.2f} (peak: ${peak_capital:.2f})")

        except Exception as e:
            print(f"Failed to save capital snapshot: {e}")

    async def load_capital_from_db(self) -> Optional[dict]:
        """
        Load last known capital state from PostgreSQL.

        Returns:
            Dict with capital info or None if no snapshot exists
        """
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(CapitalSnapshot)
                    .where(CapitalSnapshot.is_latest == True)
                    .limit(1)
                )
                snapshot = result.scalar_one_or_none()

                if snapshot:
                    return {
                        "capital": snapshot.capital,
                        "peak_capital": snapshot.peak_capital,
                        "current_drawdown_pct": snapshot.current_drawdown_pct,
                        "max_drawdown_pct": snapshot.max_drawdown_pct,
                        "timestamp": snapshot.timestamp.isoformat(),
                    }

        except Exception as e:
            print(f"Failed to load capital snapshot: {e}")

        return None

    async def initialize_capital(self, default_capital: float = 10000.0):
        """
        Initialize capital on startup.

        Loads from database if available, otherwise uses default.

        Args:
            default_capital: Default starting capital if no snapshot exists
        """
        saved = await self.load_capital_from_db()

        if saved:
            self.capital = saved["capital"]
            print(f"Loaded capital from DB: ${self.capital:.2f}")
            return saved
        else:
            self.capital = default_capital
            print(f"Using default capital: ${self.capital:.2f}")
            return None


# Global singleton
position_manager = PositionManager()
