"""SQLAlchemy database models."""

from sqlalchemy import Column, String, Float, BigInteger, DateTime, Boolean, Index
from sqlalchemy.sql import func
from datetime import datetime

from .database import Base


class Candle(Base):
    """OHLCV candle data."""
    __tablename__ = "candles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)  # e.g., "ETHUSDC"
    interval = Column(String(10), nullable=False)  # e.g., "1m", "5m"

    open_time = Column(BigInteger, nullable=False)  # Unix timestamp ms
    close_time = Column(BigInteger, nullable=False)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float, nullable=False)  # Volume in quote asset (USDC)

    num_trades = Column(BigInteger, default=0)
    is_closed = Column(Boolean, default=False)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_candles_symbol_interval_opentime", "symbol", "interval", "open_time", unique=True),
        Index("ix_candles_opentime", "open_time"),
    )


class Trade(Base):
    """Individual trade/tick data (for recent buffer)."""
    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)

    trade_id = Column(BigInteger, nullable=False)  # Binance trade ID
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # Unix timestamp ms
    is_buyer_maker = Column(Boolean, default=False)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_trades_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_trades_tradeid", "trade_id", unique=True),
    )


class Decision(Base):
    """Trading decisions for audit trail."""
    __tablename__ = "decisions"

    id = Column(String(36), primary_key=True)  # UUID
    timestamp = Column(DateTime, nullable=False, default=func.now())
    asset = Column(String(20), nullable=False)

    # Market state snapshot
    price = Column(Float, nullable=False)
    rsi = Column(Float)
    macd = Column(Float)
    volatility = Column(Float)

    # Model output
    probability_up = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    shap_values_json = Column(String(1000))  # JSON string

    # Decision
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    reason = Column(String(500))
    position_size_pct = Column(Float)

    # Risk validation
    risk_passed = Column(Boolean, nullable=False)
    risk_score = Column(Float)
    risk_violations_json = Column(String(500))

    # Execution
    execution_status = Column(String(20), default="pending")
    decision_hash = Column(String(66))  # 0x + 64 hex chars
    tx_hash = Column(String(66))

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_decisions_timestamp", "timestamp"),
        Index("ix_decisions_asset_timestamp", "asset", "timestamp"),
    )


class TradeExecution(Base):
    """Executed trades."""
    __tablename__ = "trade_executions"

    id = Column(String(36), primary_key=True)  # UUID
    decision_id = Column(String(36), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=func.now())

    asset = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    value_usd = Column(Float, nullable=False)

    status = Column(String(20), default="pending")
    tx_hash = Column(String(66))
    gas_used = Column(BigInteger)
    slippage_bps = Column(Float)

    error_message = Column(String(500))

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_trades_decision", "decision_id"),
        Index("ix_trades_timestamp", "timestamp"),
    )


class RiskSnapshot(Base):
    """Point-in-time risk state snapshots."""
    __tablename__ = "risk_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())

    total_exposure_pct = Column(Float, default=0.0)
    daily_pnl_pct = Column(Float, default=0.0)
    current_drawdown_pct = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    peak_capital = Column(Float)
    current_capital = Column(Float)  # Added for capital tracking

    trades_today = Column(BigInteger, default=0)
    trading_enabled = Column(Boolean, default=True)
    circuit_breaker_reason = Column(String(200))

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_risk_timestamp", "timestamp"),
    )


class CapitalSnapshot(Base):
    """Capital state persistence for session recovery."""
    __tablename__ = "capital_snapshots"

    id = Column(String(36), primary_key=True)  # UUID or session ID
    timestamp = Column(DateTime, nullable=False, default=func.now())

    capital = Column(Float, nullable=False)
    peak_capital = Column(Float, nullable=False)
    current_drawdown_pct = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)

    session_id = Column(String(100))  # Optional session identifier
    is_latest = Column(Boolean, default=True)  # Flag for easy lookup

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_capital_timestamp", "timestamp"),
        Index("ix_capital_latest", "is_latest"),
    )
