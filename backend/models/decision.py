"""Decision-related Pydantic models."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum
import hashlib
import json


class TradeAction(str, Enum):
    """Possible trade actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketState(BaseModel):
    """Current market state snapshot."""
    price: float
    rsi: float
    macd: float
    macd_signal: float
    ema_ratio: float
    volatility: float
    volume_spike: float
    momentum: float
    bollinger_position: float


class ModelOutput(BaseModel):
    """ML model prediction output."""
    probability_up: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    shap_values: dict[str, float] = Field(default_factory=dict)


class StrategyDecision(BaseModel):
    """Strategy agent decision."""
    action: TradeAction
    reason: str
    position_size_pct: float = Field(ge=0, le=1)


class RiskValidation(BaseModel):
    """Risk guardian validation result."""
    passed: bool
    portfolio_exposure_pct: float
    daily_pnl_pct: float
    current_drawdown_pct: float
    risk_score: float = Field(ge=0, le=1)
    violations: list[str] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    """Complete decision record for transparency."""
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    asset: str = "ETH/USDC"

    market_state: MarketState
    model_output: ModelOutput
    strategy_decision: StrategyDecision
    risk_validation: RiskValidation

    execution_status: str = "pending"
    trade_intent_hash: Optional[str] = None
    tx_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute SHA256 hash of the decision for on-chain storage."""
        # Canonical JSON representation
        data = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "market_state": self.market_state.model_dump(),
            "model_output": self.model_output.model_dump(),
            "strategy_decision": {
                "action": self.strategy_decision.action.value,
                "reason": self.strategy_decision.reason,
                "position_size_pct": self.strategy_decision.position_size_pct,
            },
            "risk_validation": self.risk_validation.model_dump(),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return "0x" + hashlib.sha256(canonical.encode()).hexdigest()


class DecisionSummary(BaseModel):
    """Lightweight decision summary for lists."""
    id: str
    timestamp: datetime
    action: TradeAction
    probability_up: float
    risk_score: float
    decision_hash: str
