"""
Risk Guardian - Validates trades against risk limits.

All trades must pass risk validation before execution.
This is a hard gate that cannot be bypassed.
"""

from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from models.risk import RiskLimits, RiskState, RiskCheckResult
from events.publisher import event_publisher
from config import get_settings

settings = get_settings()


@dataclass
class RiskConfig:
    """Risk limits configuration."""
    max_position_size_pct: float = 0.05      # 5% max position
    max_total_exposure_pct: float = 0.10     # 10% max total exposure
    max_daily_loss_pct: float = 0.03         # 3% daily loss limit
    max_drawdown_pct: float = 0.10           # 10% max drawdown
    min_trade_interval_seconds: int = 60     # 60s between trades
    max_trades_per_day: int = 50             # Max 50 trades/day
    stop_loss_pct: float = 0.02              # 2% stop loss
    take_profit_pct: float = 0.04            # 4% take profit
    max_position_age_seconds: int = 1800     # 30 minutes max hold


class RiskGuardian:
    """
    Validates all trade actions against risk limits.

    Maintains risk state in Redis and provides pass/fail
    decisions for proposed trades.
    """

    RISK_STATE_KEY = "risk:state"
    DAILY_PNL_KEY = "risk:daily_pnl"
    LAST_TRADE_KEY = "trade:last_time"
    TRADES_TODAY_KEY = "risk:trades_today"

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig(
            max_position_size_pct=settings.max_position_size,
            max_daily_loss_pct=settings.max_daily_loss,
            max_drawdown_pct=settings.max_drawdown,
            min_trade_interval_seconds=settings.trade_interval_seconds,
        )
        self._state = RiskState()
        self._circuit_breaker_active = False

    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def is_trading_enabled(self) -> bool:
        return self._state.trading_enabled and not self._circuit_breaker_active

    async def load_state(self):
        """Load risk state from Redis."""
        data = await event_publisher.get_json(self.RISK_STATE_KEY)
        if data:
            self._state = RiskState(**data)
        else:
            self._state = RiskState()

        # Load daily PnL
        daily_pnl = await event_publisher.get_json(self.DAILY_PNL_KEY)
        if daily_pnl:
            self._state.daily_pnl_pct = daily_pnl.get("value", 0.0)

        # Load trades today
        trades_today = await event_publisher.get_json(self.TRADES_TODAY_KEY)
        if trades_today:
            self._state.trades_today = trades_today.get("count", 0)

        # Load last trade time
        last_trade = await event_publisher.get_json(self.LAST_TRADE_KEY)
        if last_trade and last_trade.get("timestamp"):
            self._state.last_trade_timestamp = datetime.fromisoformat(
                last_trade["timestamp"].replace("Z", "+00:00")
            )

    async def save_state(self):
        """Save risk state to Redis."""
        await event_publisher.set_json(
            self.RISK_STATE_KEY,
            self._state.model_dump(),
        )

    async def check_trade(
        self,
        action: str,
        position_size_pct: float,
        current_exposure_pct: float,
    ) -> RiskCheckResult:
        """
        Check if a proposed trade passes all risk checks.

        Args:
            action: "BUY" or "SELL"
            position_size_pct: Size of proposed position as % of capital
            current_exposure_pct: Current portfolio exposure

        Returns:
            RiskCheckResult with pass/fail and details
        """
        checks = {}
        violations = []
        warnings = []

        # 1. Circuit breaker check
        checks["circuit_breaker"] = not self._circuit_breaker_active
        if self._circuit_breaker_active:
            violations.append(f"Circuit breaker active: {self._state.circuit_breaker_reason}")

        # 2. Trading enabled check
        checks["trading_enabled"] = self._state.trading_enabled
        if not self._state.trading_enabled:
            violations.append("Trading is disabled")

        # 3. Position size check
        checks["position_size"] = position_size_pct <= self.config.max_position_size_pct
        if not checks["position_size"]:
            violations.append(
                f"Position size {position_size_pct:.1%} exceeds limit {self.config.max_position_size_pct:.1%}"
            )

        # 4. Total exposure check (for BUY only)
        if action == "BUY":
            new_exposure = current_exposure_pct + position_size_pct
            checks["total_exposure"] = new_exposure <= self.config.max_total_exposure_pct
            if not checks["total_exposure"]:
                violations.append(
                    f"New exposure {new_exposure:.1%} would exceed limit {self.config.max_total_exposure_pct:.1%}"
                )
        else:
            checks["total_exposure"] = True

        # 5. Daily loss check
        checks["daily_loss"] = self._state.daily_pnl_pct >= -self.config.max_daily_loss_pct
        if not checks["daily_loss"]:
            violations.append(
                f"Daily loss {self._state.daily_pnl_pct:.1%} exceeds limit -{self.config.max_daily_loss_pct:.1%}"
            )

        # 6. Drawdown check
        checks["drawdown"] = self._state.current_drawdown_pct <= self.config.max_drawdown_pct
        if not checks["drawdown"]:
            violations.append(
                f"Drawdown {self._state.current_drawdown_pct:.1%} exceeds limit {self.config.max_drawdown_pct:.1%}"
            )

        # 7. Trade cooldown check
        checks["cooldown"] = self._check_cooldown()
        if not checks["cooldown"]:
            violations.append(
                f"Trade cooldown not met ({self.config.min_trade_interval_seconds}s required)"
            )

        # 8. Daily trade count check
        checks["trade_count"] = self._state.trades_today < self.config.max_trades_per_day
        if not checks["trade_count"]:
            violations.append(
                f"Daily trade limit reached ({self.config.max_trades_per_day})"
            )

        # Calculate risk score (0 = no risk, 1 = max risk)
        risk_factors = [
            current_exposure_pct / self.config.max_total_exposure_pct,
            abs(self._state.daily_pnl_pct) / self.config.max_daily_loss_pct,
            self._state.current_drawdown_pct / self.config.max_drawdown_pct,
            self._state.trades_today / self.config.max_trades_per_day,
        ]
        risk_score = min(1.0, max(risk_factors))

        # Add warnings for elevated risk
        if risk_score > 0.7:
            warnings.append(f"Elevated risk score: {risk_score:.2f}")
        if self._state.daily_pnl_pct < -0.02:
            warnings.append(f"Daily PnL nearing limit: {self._state.daily_pnl_pct:.1%}")

        # All checks must pass
        can_trade = all(checks.values())

        if can_trade:
            return RiskCheckResult.passed(risk_score, checks)
        else:
            result = RiskCheckResult.failed(violations, checks)
            result.warnings = warnings
            return result

    def _check_cooldown(self) -> bool:
        """Check if enough time has passed since last trade."""
        if self._state.last_trade_timestamp is None:
            return True

        now = datetime.now(timezone.utc)
        last = self._state.last_trade_timestamp

        # Handle timezone-naive timestamps
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)

        elapsed = (now - last).total_seconds()
        return elapsed >= self.config.min_trade_interval_seconds

    def check_stop_loss(self, unrealized_pnl_pct: float) -> bool:
        """Check if position should be stopped out."""
        return unrealized_pnl_pct <= -self.config.stop_loss_pct

    def check_take_profit(self, unrealized_pnl_pct: float) -> bool:
        """Check if position should take profit."""
        return unrealized_pnl_pct >= self.config.take_profit_pct

    def check_position_age(self, age_seconds: float) -> bool:
        """Check if position has exceeded max hold time."""
        return age_seconds >= self.config.max_position_age_seconds

    async def record_trade(self, pnl: float = 0.0):
        """
        Record that a trade was executed.

        Updates:
        - Last trade timestamp
        - Trades today count
        - Daily PnL if closing position
        """
        now = datetime.now(timezone.utc)

        # Update last trade time
        self._state.last_trade_timestamp = now
        await event_publisher.set_json(
            self.LAST_TRADE_KEY,
            {"timestamp": now.isoformat()},
            expire_seconds=86400,
        )

        # Increment trades today
        self._state.trades_today += 1
        await event_publisher.set_json(
            self.TRADES_TODAY_KEY,
            {"count": self._state.trades_today, "date": now.date().isoformat()},
            expire_seconds=86400,
        )

        # Update daily PnL if trade closed with PnL
        if pnl != 0.0:
            self._state.daily_pnl_pct += pnl
            await event_publisher.set_json(
                self.DAILY_PNL_KEY,
                {"value": self._state.daily_pnl_pct, "date": now.date().isoformat()},
                expire_seconds=86400,
            )

        await self.save_state()

    async def trigger_circuit_breaker(self, reason: str):
        """
        Activate circuit breaker - stops all trading.

        Args:
            reason: Why the circuit breaker was triggered
        """
        self._circuit_breaker_active = True
        self._state.trading_enabled = False
        self._state.circuit_breaker_reason = reason
        await self.save_state()
        print(f"CIRCUIT BREAKER TRIGGERED: {reason}")

    async def reset_circuit_breaker(self):
        """Reset circuit breaker (manual only)."""
        self._circuit_breaker_active = False
        self._state.trading_enabled = True
        self._state.circuit_breaker_reason = None
        await self.save_state()
        print("Circuit breaker reset")

    def get_config(self) -> dict:
        """Get current risk configuration."""
        return {
            "max_position_size_pct": self.config.max_position_size_pct,
            "max_total_exposure_pct": self.config.max_total_exposure_pct,
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "min_trade_interval_seconds": self.config.min_trade_interval_seconds,
            "max_trades_per_day": self.config.max_trades_per_day,
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "max_position_age_seconds": self.config.max_position_age_seconds,
        }


# Global singleton
risk_guardian = RiskGuardian()
