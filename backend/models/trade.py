"""Trade-related Pydantic models."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum


class TradeStatus(str, Enum):
    """Trade execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TradeIntent(BaseModel):
    """EIP-712 compatible trade intent."""
    agent_address: str
    asset: str = "ETH"
    action: str  # BUY or SELL
    amount: str  # Amount in wei/units as string
    max_slippage_bps: int = 50  # 0.5% = 50 basis points
    deadline: int  # Unix timestamp
    decision_hash: str
    nonce: int

    def to_eip712_message(self) -> dict:
        """Convert to EIP-712 typed data structure."""
        return {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "TradeIntent": [
                    {"name": "agent", "type": "address"},
                    {"name": "asset", "type": "string"},
                    {"name": "action", "type": "string"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "maxSlippageBps", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                    {"name": "decisionHash", "type": "bytes32"},
                    {"name": "nonce", "type": "uint256"},
                ],
            },
            "primaryType": "TradeIntent",
            "domain": {
                "name": "VAPM Trade Executor",
                "version": "1",
                "chainId": 84532,  # Base Sepolia
                "verifyingContract": "0x...",  # TradeExecutor address
            },
            "message": {
                "agent": self.agent_address,
                "asset": self.asset,
                "action": self.action,
                "amount": int(self.amount),
                "maxSlippageBps": self.max_slippage_bps,
                "deadline": self.deadline,
                "decisionHash": self.decision_hash,
                "nonce": self.nonce,
            },
        }


class Trade(BaseModel):
    """Executed trade record."""
    id: str
    decision_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    asset: str
    action: str
    amount: float
    price: float
    value_usd: float

    status: TradeStatus = TradeStatus.PENDING
    tx_hash: Optional[str] = None
    gas_used: Optional[int] = None
    slippage_actual_bps: Optional[int] = None

    error_message: Optional[str] = None


class Position(BaseModel):
    """Current position state."""
    asset: str
    size: float  # Amount of asset
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    @property
    def value_usd(self) -> float:
        return self.size * self.current_price


class Portfolio(BaseModel):
    """Portfolio state."""
    total_capital: float
    available_capital: float
    positions: list[Position] = Field(default_factory=list)
    total_pnl: float = 0.0
    daily_pnl: float = 0.0

    @property
    def exposure_pct(self) -> float:
        """Calculate current exposure as percentage of capital."""
        position_value = sum(p.value_usd for p in self.positions)
        return position_value / self.total_capital if self.total_capital > 0 else 0
