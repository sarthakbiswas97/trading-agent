"""
Pure technical indicator functions - no external dependencies.

This module contains only pure mathematical functions for computing
technical indicators. It has no async code, no Redis, no database -
just numpy operations.

Used by:
- backend/services/feature_engine.py (real-time features)
- ml/data_preparation.py (training data generation)
"""

import numpy as np


# ============================================
# CONSTANTS
# ============================================

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

# Minimum candles needed for feature computation
MIN_CANDLES = MACD_SLOW + MACD_SIGNAL  # 35


# ============================================
# HELPER FUNCTIONS
# ============================================

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Exponential Moving Average.

    Args:
        data: Input array of values
        period: EMA period (e.g., 14 for RSI)

    Returns:
        Array of EMA values (same length as input)
    """
    alpha = 2 / (period + 1)
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]

    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


# ============================================
# INDICATOR FUNCTIONS
# ============================================

def compute_rsi(closes: np.ndarray) -> float:
    """
    Compute Relative Strength Index (0-100).

    RSI measures momentum - values below 30 indicate oversold,
    above 70 indicate overbought.

    Args:
        closes: Array of closing prices

    Returns:
        RSI value between 0 and 100
    """
    if len(closes) < RSI_PERIOD + 1:
        return 50.0  # Neutral when not enough data

    # Calculate price changes
    deltas = np.diff(closes)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use EMA for smoothing
    avg_gain = ema(gains, RSI_PERIOD)[-1]
    avg_loss = ema(losses, RSI_PERIOD)[-1]

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def compute_macd(closes: np.ndarray) -> tuple[float, float, float]:
    """
    Compute MACD, Signal line, and Histogram.

    MACD shows trend direction and momentum. Crossovers between
    MACD and Signal line indicate potential trade signals.

    Args:
        closes: Array of closing prices

    Returns:
        Tuple of (macd, signal, histogram) - normalized by price
    """
    if len(closes) < MACD_SLOW + MACD_SIGNAL:
        return 0.0, 0.0, 0.0

    # Compute fast and slow EMAs
    ema_fast = ema(closes, MACD_FAST)
    ema_slow = ema(closes, MACD_SLOW)

    # MACD line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow

    # Signal line = EMA of MACD line
    signal_line = ema(macd_line, MACD_SIGNAL)

    # Histogram = MACD - Signal
    histogram = macd_line - signal_line

    # Normalize by price to make comparable across different price levels
    price = closes[-1]
    return (
        float(macd_line[-1] / price),
        float(signal_line[-1] / price),
        float(histogram[-1] / price),
    )


def compute_ema_ratio(closes: np.ndarray) -> float:
    """
    Compute price / EMA ratio.

    Values > 1 mean price is above trend (bullish).
    Values < 1 mean price is below trend (bearish).

    Args:
        closes: Array of closing prices

    Returns:
        Ratio of current price to EMA (typically 0.95 - 1.05)
    """
    if len(closes) < EMA_PERIOD:
        return 1.0

    ema_value = ema(closes, EMA_PERIOD)[-1]
    return float(closes[-1] / ema_value)


def compute_volatility(closes: np.ndarray) -> float:
    """
    Compute rolling volatility (standard deviation of returns).

    Higher volatility = more risk/opportunity.
    Typical values: 0.001 - 0.05

    Args:
        closes: Array of closing prices

    Returns:
        Standard deviation of recent returns
    """
    if len(closes) < VOLATILITY_PERIOD + 1:
        return 0.0

    # Calculate returns (percentage change)
    returns = np.diff(closes) / closes[:-1]

    # Standard deviation of recent returns
    volatility = np.std(returns[-VOLATILITY_PERIOD:])

    return float(volatility)


def compute_volume_spike(volumes: np.ndarray) -> float:
    """
    Compute volume spike ratio (current volume / average volume).

    Values > 1.5 indicate unusual activity.
    Values < 0.5 indicate low activity.

    Args:
        volumes: Array of volume values

    Returns:
        Ratio of current volume to average (typically 0.2 - 3.0)
    """
    if len(volumes) < VOLUME_AVG_PERIOD:
        return 1.0

    # Average of previous volumes (excluding current)
    avg_volume = np.mean(volumes[-VOLUME_AVG_PERIOD - 1:-1])

    if avg_volume == 0:
        return 1.0

    return float(volumes[-1] / avg_volume)


def compute_momentum(closes: np.ndarray) -> float:
    """
    Compute momentum (rate of change over N periods).

    Positive = price going up
    Negative = price going down

    Args:
        closes: Array of closing prices

    Returns:
        Rate of change (typically -0.1 to 0.1)
    """
    if len(closes) < MOMENTUM_PERIOD + 1:
        return 0.0

    past_price = closes[-MOMENTUM_PERIOD - 1]
    current_price = closes[-1]

    momentum = (current_price - past_price) / past_price

    return float(momentum)


def compute_bollinger_position(closes: np.ndarray) -> float:
    """
    Compute position within Bollinger Bands (-1 to 1).

    -1 = at lower band (potentially oversold)
    +1 = at upper band (potentially overbought)
     0 = at middle (SMA)

    Args:
        closes: Array of closing prices

    Returns:
        Position within bands, clipped to [-1, 1]
    """
    if len(closes) < BOLLINGER_PERIOD:
        return 0.0

    # Simple Moving Average
    sma = np.mean(closes[-BOLLINGER_PERIOD:])

    # Standard deviation
    std = np.std(closes[-BOLLINGER_PERIOD:])

    if std == 0:
        return 0.0

    # Bollinger Bands
    upper_band = sma + BOLLINGER_STD * std
    lower_band = sma - BOLLINGER_STD * std

    band_width = upper_band - lower_band
    if band_width == 0:
        return 0.0

    # Normalize to -1 to 1
    position = (closes[-1] - lower_band) / band_width * 2 - 1

    return float(np.clip(position, -1, 1))


# ============================================
# MAIN FUNCTION - Compute all features at once
# ============================================

def compute_all_features(closes: np.ndarray, volumes: np.ndarray) -> dict:
    """
    Compute all technical indicators from price and volume arrays.

    This is the main function used for both real-time inference
    and training data generation.

    Args:
        closes: Array of closing prices (need at least MIN_CANDLES)
        volumes: Array of volumes (same length as closes)

    Returns:
        Dictionary with all computed features

    Raises:
        ValueError: If not enough candles provided
    """
    if len(closes) < MIN_CANDLES:
        raise ValueError(f"Need at least {MIN_CANDLES} candles, got {len(closes)}")

    if len(closes) != len(volumes):
        raise ValueError(f"closes and volumes must have same length")

    # Compute MACD (returns 3 values)
    macd, macd_signal, macd_histogram = compute_macd(closes)

    return {
        "price": float(closes[-1]),
        "rsi": compute_rsi(closes),
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "ema_ratio": compute_ema_ratio(closes),
        "volatility": compute_volatility(closes),
        "volume_spike": compute_volume_spike(volumes),
        "momentum": compute_momentum(closes),
        "bollinger_position": compute_bollinger_position(closes),
    }


def normalize_features(features: dict) -> np.ndarray:
    """
    Convert feature dict to normalized numpy array for ML model.

    This applies the same normalization as FeatureVector.to_array()

    Args:
        features: Dictionary from compute_all_features()

    Returns:
        Numpy array of 9 normalized features
    """
    return np.array([
        features["rsi"] / 100,              # Normalize to 0-1
        features["macd"],                    # Already normalized by price
        features["macd_signal"],
        features["macd_histogram"],
        features["ema_ratio"] - 1,           # Center around 0
        features["volatility"],
        features["volume_spike"] - 1,        # Center around 0
        features["momentum"],
        features["bollinger_position"],
    ])


# ============================================
# FEATURE NAMES (for reference)
# ============================================

FEATURE_NAMES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "ema_ratio",
    "volatility",
    "volume_spike",
    "momentum",
    "bollinger_position",
]
