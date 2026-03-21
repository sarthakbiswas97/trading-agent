"""
Feature engine - computes technical indicators from candle data.

This service handles:
- Fetching candles from market data service
- Computing features using shared indicator functions
- Caching results in Redis
- Publishing events for other services

The actual indicator computations are in core/indicators.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .market_data import CandleData, market_data_service
from events.publisher import event_publisher
from core.indicators import (
    compute_all_features,
    normalize_features,
    MIN_CANDLES,
    FEATURE_NAMES,
)


@dataclass
class FeatureVector:
    """Technical indicators for ML model input."""
    timestamp: int
    price: float

    # Base indicators
    rsi: float                 # Relative Strength Index (0-100)
    macd: float                # MACD line
    macd_signal: float         # MACD signal line
    macd_histogram: float      # MACD histogram
    ema_ratio: float           # Price / EMA20 ratio
    volatility: float          # Rolling std dev of returns
    volume_spike: float        # Volume / avg volume ratio
    momentum: float            # Price change over N periods
    bollinger_position: float  # Position within Bollinger bands (-1 to 1)

    # Regime indicators (for risk management)
    adx: float = 25.0              # Average Directional Index (trend strength)
    atr: float = 0.0               # Average True Range (absolute volatility)
    volatility_regime: float = 0.5 # Volatility percentile (0-1)
    price_acceleration: float = 0.0  # 2nd derivative of price
    range_position: float = 0.0    # Position in recent range (-1 to 1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model (base features only)."""
        return np.array([
            self.rsi / 100,              # Normalize to 0-1
            self.macd,                   # Already normalized by price scale
            self.macd_signal,
            self.macd_histogram,
            self.ema_ratio - 1,          # Center around 0
            self.volatility,
            self.volume_spike - 1,       # Center around 0
            self.momentum,
            self.bollinger_position,
        ])

    def to_array_full(self) -> np.ndarray:
        """Convert to numpy array including regime features (14 features)."""
        return np.array([
            self.rsi / 100,
            self.macd,
            self.macd_signal,
            self.macd_histogram,
            self.ema_ratio - 1,
            self.volatility,
            self.volume_spike - 1,
            self.momentum,
            self.bollinger_position,
            self.adx / 100,              # Normalize to 0-1
            self.atr / self.price,       # Normalize by price
            self.volatility_regime,      # Already 0-1
            self.price_acceleration * 100,  # Scale up
            self.range_position,         # Already -1 to 1
        ])

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "rsi": round(self.rsi, 2),
            "macd": round(self.macd, 6),
            "macd_signal": round(self.macd_signal, 6),
            "macd_histogram": round(self.macd_histogram, 6),
            "ema_ratio": round(self.ema_ratio, 4),
            "volatility": round(self.volatility, 6),
            "volume_spike": round(self.volume_spike, 4),
            "momentum": round(self.momentum, 6),
            "bollinger_position": round(self.bollinger_position, 4),
            # Regime indicators
            "adx": round(self.adx, 2),
            "atr": round(self.atr, 4),
            "volatility_regime": round(self.volatility_regime, 4),
            "price_acceleration": round(self.price_acceleration, 6),
            "range_position": round(self.range_position, 4),
        }

    @classmethod
    def from_dict(cls, data: dict, timestamp: int) -> "FeatureVector":
        """Create FeatureVector from compute_all_features output."""
        return cls(
            timestamp=timestamp,
            price=data["price"],
            rsi=data["rsi"],
            macd=data["macd"],
            macd_signal=data["macd_signal"],
            macd_histogram=data["macd_histogram"],
            ema_ratio=data["ema_ratio"],
            volatility=data["volatility"],
            volume_spike=data["volume_spike"],
            momentum=data["momentum"],
            bollinger_position=data["bollinger_position"],
            # Regime indicators (with defaults for backward compatibility)
            adx=data.get("adx", 25.0),
            atr=data.get("atr", 0.0),
            volatility_regime=data.get("volatility_regime", 0.5),
            price_acceleration=data.get("price_acceleration", 0.0),
            range_position=data.get("range_position", 0.0),
        )


class FeatureEngine:
    """
    Computes technical indicators from candle data.

    This class handles the async operations (fetching candles, Redis caching).
    The actual indicator math is delegated to core.indicators module.
    """

    def __init__(self):
        self.latest_features: Optional[FeatureVector] = None

    async def compute_features(self, candles: list[CandleData] = None) -> Optional[FeatureVector]:
        """
        Compute features from candle data.

        Args:
            candles: List of CandleData objects. If None, fetches from market_data_service.

        Returns:
            FeatureVector with all computed indicators, or None if not enough data.
        """
        # Get candles if not provided
        if candles is None:
            candles = await market_data_service.get_recent_candles(limit=150)

        if len(candles) < MIN_CANDLES:
            print(f"Not enough candles: {len(candles)} < {MIN_CANDLES}")
            return None

        # Extract OHLCV arrays
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Compute all indicators including regime features (ADX, ATR, etc.)
        features_dict = compute_all_features(
            closes, volumes, highs, lows,
            include_regime=True
        )

        # Create feature vector
        features = FeatureVector.from_dict(
            features_dict,
            timestamp=candles[-1].close_time,
        )

        self.latest_features = features

        # Store in Redis for other services
        await event_publisher.set_json(
            "features:latest",
            features.to_dict(),
            expire_seconds=120,
        )

        # Publish event
        await event_publisher.publish(
            "event:features_computed",
            features.to_dict(),
        )

        return features


# Global singleton
feature_engine = FeatureEngine()
