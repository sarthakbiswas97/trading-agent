"""Feature engine - computes technical indicators from candle data."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .market_data import CandleData, market_data_service
from events.publisher import event_publisher


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


class FeatureEngine:
    """Computes technical indicators from candle data."""

    # Indicator parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    EMA_PERIOD = 20
    VOLATILITY_PERIOD = 20
    VOLUME_AVG_PERIOD = 20
    MOMENTUM_PERIOD = 10
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2

    def __init__(self):
        self.latest_features: Optional[FeatureVector] = None

    async def compute_features(self, candles: list[CandleData] = None) -> Optional[FeatureVector]:
        """Compute features from candle data."""
        # Get candles if not provided
        if candles is None:
            candles = await market_data_service.get_recent_candles(limit=100)

        if len(candles) < self.MACD_SLOW + self.MACD_SIGNAL:
            print(f"Not enough candles: {len(candles)} < {self.MACD_SLOW + self.MACD_SIGNAL}")
            return None

        # Extract price and volume arrays
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Compute all indicators
        rsi = self._compute_rsi(closes)
        macd, macd_signal, macd_histogram = self._compute_macd(closes)
        ema_ratio = self._compute_ema_ratio(closes)
        volatility = self._compute_volatility(closes)
        volume_spike = self._compute_volume_spike(volumes)
        momentum = self._compute_momentum(closes)
        bollinger_position = self._compute_bollinger_position(closes)

        # Create feature vector
        features = FeatureVector(
            timestamp=candles[-1].close_time,
            price=closes[-1],
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            ema_ratio=ema_ratio,
            volatility=volatility,
            volume_spike=volume_spike,
            momentum=momentum,
            bollinger_position=bollinger_position,
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

    def _compute_rsi(self, closes: np.ndarray) -> float:
        """Compute Relative Strength Index."""
        if len(closes) < self.RSI_PERIOD + 1:
            return 50.0  # Neutral

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use exponential moving average for smoothing
        avg_gain = self._ema(gains, self.RSI_PERIOD)[-1]
        avg_loss = self._ema(losses, self.RSI_PERIOD)[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _compute_macd(self, closes: np.ndarray) -> tuple[float, float, float]:
        """Compute MACD, Signal, and Histogram."""
        if len(closes) < self.MACD_SLOW + self.MACD_SIGNAL:
            return 0.0, 0.0, 0.0

        ema_fast = self._ema(closes, self.MACD_FAST)
        ema_slow = self._ema(closes, self.MACD_SLOW)

        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.MACD_SIGNAL)
        histogram = macd_line - signal_line

        # Normalize by price to make it comparable across different price levels
        price = closes[-1]
        return (
            float(macd_line[-1] / price),
            float(signal_line[-1] / price),
            float(histogram[-1] / price),
        )

    def _compute_ema_ratio(self, closes: np.ndarray) -> float:
        """Compute price / EMA ratio."""
        if len(closes) < self.EMA_PERIOD:
            return 1.0

        ema = self._ema(closes, self.EMA_PERIOD)[-1]
        return float(closes[-1] / ema)

    def _compute_volatility(self, closes: np.ndarray) -> float:
        """Compute rolling volatility (standard deviation of returns)."""
        if len(closes) < self.VOLATILITY_PERIOD + 1:
            return 0.0

        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-self.VOLATILITY_PERIOD:])

        return float(volatility)

    def _compute_volume_spike(self, volumes: np.ndarray) -> float:
        """Compute volume spike ratio (current / average)."""
        if len(volumes) < self.VOLUME_AVG_PERIOD:
            return 1.0

        avg_volume = np.mean(volumes[-self.VOLUME_AVG_PERIOD:-1])
        if avg_volume == 0:
            return 1.0

        return float(volumes[-1] / avg_volume)

    def _compute_momentum(self, closes: np.ndarray) -> float:
        """Compute momentum (rate of change)."""
        if len(closes) < self.MOMENTUM_PERIOD + 1:
            return 0.0

        momentum = (closes[-1] - closes[-self.MOMENTUM_PERIOD - 1]) / closes[-self.MOMENTUM_PERIOD - 1]
        return float(momentum)

    def _compute_bollinger_position(self, closes: np.ndarray) -> float:
        """Compute position within Bollinger Bands (-1 to 1)."""
        if len(closes) < self.BOLLINGER_PERIOD:
            return 0.0

        sma = np.mean(closes[-self.BOLLINGER_PERIOD:])
        std = np.std(closes[-self.BOLLINGER_PERIOD:])

        if std == 0:
            return 0.0

        upper_band = sma + self.BOLLINGER_STD * std
        lower_band = sma - self.BOLLINGER_STD * std

        # Normalize to -1 (at lower band) to 1 (at upper band)
        band_width = upper_band - lower_band
        if band_width == 0:
            return 0.0

        position = (closes[-1] - lower_band) / band_width * 2 - 1
        return float(np.clip(position, -1, 1))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema


# Global singleton
feature_engine = FeatureEngine()
