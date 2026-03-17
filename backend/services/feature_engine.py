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

    # Indicators
    rsi: float                 # Relative Strength Index (0-100)
    macd: float                # MACD line
    macd_signal: float         # MACD signal line
    macd_histogram: float      # MACD histogram
    ema_ratio: float           # Price / EMA20 ratio
    volatility: float          # Rolling std dev of returns
    volume_spike: float        # Volume / avg volume ratio
    momentum: float            # Price change over N periods
    bollinger_position: float  # Position within Bollinger bands (-1 to 1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
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
            candles = await market_data_service.get_recent_candles(limit=100)

        if len(candles) < MIN_CANDLES:
            print(f"Not enough candles: {len(candles)} < {MIN_CANDLES}")
            return None

        # Extract price and volume arrays
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Compute all indicators using shared function
        features_dict = compute_all_features(closes, volumes)

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
