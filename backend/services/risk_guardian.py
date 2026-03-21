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
    circuit_breaker_drawdown_pct: float = 0.08  # 8% auto circuit breaker
    min_trade_interval_seconds: int = 60     # 60s between trades
    max_trades_per_day: int = 50             # Max 50 trades/day
    stop_loss_pct: float = 0.02              # 2% stop loss (fallback)
    take_profit_pct: float = 0.04            # 4% take profit
    max_position_age_seconds: int = 1800     # 30 minutes max hold
    target_volatility: float = 0.02          # 2% target daily volatility
    atr_stop_multiplier: float = 2.0         # ATR multiplier for stop-loss
    max_stop_loss_pct: float = 0.03          # 3% max stop-loss cap


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

    # ================================================================
    # ADAPTIVE RISK MANAGEMENT (New Methods)
    # ================================================================

    async def update_equity(self, current_capital: float):
        """
        Update equity curve and calculate drawdown.

        Called after every trade close or periodic equity check.
        Automatically triggers circuit breaker at 8% drawdown.

        Args:
            current_capital: Current portfolio value in USD
        """
        # Initialize peak capital if not set
        if self._state.peak_capital <= 0:
            self._state.peak_capital = current_capital

        # Update peak capital (high water mark)
        if current_capital > self._state.peak_capital:
            self._state.peak_capital = current_capital

        # Calculate current drawdown from peak
        if self._state.peak_capital > 0:
            self._state.current_drawdown_pct = (
                (self._state.peak_capital - current_capital) / self._state.peak_capital
            )
            # Track max drawdown ever
            self._state.max_drawdown_pct = max(
                self._state.max_drawdown_pct,
                self._state.current_drawdown_pct
            )

        # Auto circuit breaker at 8% drawdown
        if self._state.current_drawdown_pct >= self.config.circuit_breaker_drawdown_pct:
            if not self._circuit_breaker_active:
                await self.trigger_circuit_breaker(
                    f"Drawdown {self._state.current_drawdown_pct:.1%} exceeded "
                    f"{self.config.circuit_breaker_drawdown_pct:.0%} limit"
                )

        await self.save_state()

    def get_throttle_factor(self) -> float:
        """
        Calculate position size throttle based on current drawdown.

        Reduces position sizes as losses accumulate to preserve capital.

        Returns:
            Factor 0.0-1.0 to multiply position size by.

        Drawdown Brackets:
            | Drawdown | Factor | Effect        |
            |----------|--------|---------------|
            | 0-2%     | 1.00   | Full size     |
            | 2-4%     | 0.75   | 75% size      |
            | 4-6%     | 0.50   | 50% size      |
            | 6-8%     | 0.25   | 25% size      |
            | 8%+      | 0.00   | No trading    |
        """
        dd = self._state.current_drawdown_pct

        if dd < 0.02:
            return 1.0
        elif dd < 0.04:
            return 0.75
        elif dd < 0.06:
            return 0.50
        elif dd < 0.08:
            return 0.25
        else:
            return 0.0  # Circuit breaker territory

    def calculate_position_size(
        self,
        base_size_pct: float,
        current_volatility: float,
    ) -> float:
        """
        Calculate volatility-scaled position size with drawdown throttling.

        Scales position inversely with volatility:
        - High volatility → smaller position (less risk per trade)
        - Low volatility → larger position (up to 2x base)

        Also applies drawdown-based throttling.

        Args:
            base_size_pct: Base position size as % of capital (e.g., 0.03 for 3%)
            current_volatility: Current market volatility (std dev of returns)

        Returns:
            Adjusted position size as % of capital
        """
        # Volatility scaling (inverse relationship)
        # High vol = smaller position, low vol = larger position
        vol_scalar = self.config.target_volatility / max(current_volatility, 0.005)
        vol_scalar = min(max(vol_scalar, 0.5), 2.0)  # Clamp 0.5x to 2x

        # Apply drawdown throttle
        throttle = self.get_throttle_factor()

        # Calculate final size
        adjusted_size = base_size_pct * vol_scalar * throttle

        # Never exceed max position limit
        return min(adjusted_size, self.config.max_position_size_pct)

    def calculate_stop_loss_price(
        self,
        entry_price: float,
        atr: float,
    ) -> float:
        """
        Calculate dynamic stop-loss price based on ATR.

        Uses ATR × multiplier for stop distance, but caps at max percentage.

        Args:
            entry_price: Position entry price
            atr: Current Average True Range value

        Returns:
            Stop-loss price (for long positions)
        """
        # ATR-based stop distance
        atr_stop_distance = atr * self.config.atr_stop_multiplier

        # Cap at max stop percentage (e.g., 3% of entry)
        max_stop_distance = entry_price * self.config.max_stop_loss_pct

        # Use the smaller of the two (tighter stop)
        stop_distance = min(atr_stop_distance, max_stop_distance)

        return entry_price - stop_distance

    def check_stop_loss_dynamic(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
    ) -> bool:
        """
        Check if position should be stopped out using dynamic ATR-based stop.

        Args:
            entry_price: Position entry price
            current_price: Current market price
            atr: Current Average True Range

        Returns:
            True if stop-loss triggered
        """
        stop_price = self.calculate_stop_loss_price(entry_price, atr)
        return current_price <= stop_price

    def get_risk_status(self) -> dict:
        """Get comprehensive risk status for monitoring."""
        return {
            "drawdown": {
                "current_pct": round(self._state.current_drawdown_pct * 100, 2),
                "max_pct": round(self._state.max_drawdown_pct * 100, 2),
                "peak_capital": round(self._state.peak_capital, 2),
            },
            "throttle": {
                "factor": self.get_throttle_factor(),
                "reason": self._get_throttle_reason(),
            },
            "circuit_breaker": {
                "active": self._circuit_breaker_active,
                "reason": self._state.circuit_breaker_reason,
            },
            "daily": {
                "pnl_pct": round(self._state.daily_pnl_pct * 100, 2),
                "trades": self._state.trades_today,
            },
        }

    def _get_throttle_reason(self) -> str:
        """Get human-readable throttle reason."""
        dd = self._state.current_drawdown_pct
        throttle = self.get_throttle_factor()

        if throttle == 1.0:
            return "No throttling"
        elif throttle == 0.0:
            return f"Trading halted (drawdown {dd:.1%})"
        else:
            return f"Position size at {throttle:.0%} (drawdown {dd:.1%})"

    async def record_trade(self, pnl: float = 0.0, current_capital: float = None):
        """
        Record that a trade was executed.

        Updates:
        - Last trade timestamp
        - Trades today count
        - Daily PnL if closing position
        - Equity/drawdown tracking if capital provided

        Args:
            pnl: Realized PnL as percentage of capital (e.g., 0.01 = 1%)
            current_capital: Current portfolio value (for drawdown tracking)
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

        # Update equity curve and drawdown if capital provided
        if current_capital is not None:
            await self.update_equity(current_capital)
        else:
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
            "circuit_breaker_drawdown_pct": self.config.circuit_breaker_drawdown_pct,
            "min_trade_interval_seconds": self.config.min_trade_interval_seconds,
            "max_trades_per_day": self.config.max_trades_per_day,
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "max_position_age_seconds": self.config.max_position_age_seconds,
            "target_volatility": self.config.target_volatility,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "max_stop_loss_pct": self.config.max_stop_loss_pct,
        }


# Global singleton
risk_guardian = RiskGuardian()
