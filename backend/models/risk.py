"""Risk management Pydantic models."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class RiskLimits(BaseModel):
    """Hard risk limits that cannot be violated."""
    max_position_size_pct: float = 0.05      # 5% of capital per position
    max_daily_loss_pct: float = 0.03         # 3% daily loss limit
    max_drawdown_pct: float = 0.10           # 10% max drawdown
    min_trade_interval_seconds: int = 60      # 1 minute between trades
    max_trades_per_day: int = 50              # Max 50 trades per day


class RiskState(BaseModel):
    """Current risk state snapshot."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Position exposure
    total_exposure_pct: float = 0.0
    largest_position_pct: float = 0.0

    # PnL tracking
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0

    # Drawdown
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_capital: float = 0.0

    # Activity
    trades_today: int = 0
    last_trade_timestamp: Optional[datetime] = None

    # Circuit breaker
    trading_enabled: bool = True
    circuit_breaker_reason: Optional[str] = None


class RiskCheckResult(BaseModel):
    """Result of a risk check for a proposed trade."""
    can_trade: bool
    risk_score: float = Field(ge=0, le=1)
    checks: dict[str, bool] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def passed(cls, risk_score: float, checks: dict) -> "RiskCheckResult":
        """Create a passing result."""
        return cls(
            can_trade=True,
            risk_score=risk_score,
            checks=checks,
            violations=[],
        )

    @classmethod
    def failed(cls, violations: list[str], checks: dict) -> "RiskCheckResult":
        """Create a failing result."""
        return cls(
            can_trade=False,
            risk_score=1.0,
            checks=checks,
            violations=violations,
        )


class RiskMetrics(BaseModel):
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward_ratio: float = 0.0


class AgentReputation(BaseModel):
    """On-chain reputation score components."""
    overall_score: float = Field(ge=0, le=100)

    # Component scores (0-100)
    performance_score: float = 50.0
    risk_management_score: float = 50.0
    validation_score: float = 50.0
    consistency_score: float = 50.0

    # Raw metrics
    total_trades: int = 0
    profitable_trades: int = 0
    decisions_validated: int = 0
    risk_violations: int = 0

    def compute_overall(self) -> float:
        """Compute weighted overall score."""
        self.overall_score = (
            0.35 * self.performance_score +
            0.25 * self.risk_management_score +
            0.25 * self.validation_score +
            0.15 * self.consistency_score
        )
        return self.overall_score
