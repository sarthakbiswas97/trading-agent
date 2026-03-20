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

# Regime detection periods
ADX_PERIOD = 14
ATR_PERIOD = 14
VOLATILITY_REGIME_LOOKBACK = 100

# Minimum candles needed for feature computation
MIN_CANDLES = max(MACD_SLOW + MACD_SIGNAL, VOLATILITY_REGIME_LOOKBACK)  # 100


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
# REGIME INDICATORS
# ============================================

def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """
    Compute Average True Range (ATR).

    ATR measures volatility using high/low/close data.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices

    Returns:
        ATR value (absolute, not percentage)
    """
    if len(closes) < ATR_PERIOD + 1:
        return 0.0

    # True Range = max of:
    # 1. High - Low
    # 2. |High - Previous Close|
    # 3. |Low - Previous Close|
    prev_closes = closes[:-1]
    curr_highs = highs[1:]
    curr_lows = lows[1:]

    tr1 = curr_highs - curr_lows
    tr2 = np.abs(curr_highs - prev_closes)
    tr3 = np.abs(curr_lows - prev_closes)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # ATR is EMA of True Range
    atr_values = ema(true_range, ATR_PERIOD)

    return float(atr_values[-1])


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """
    Compute Average Directional Index (ADX).

    ADX measures trend strength (not direction):
    - 0-20: Weak/no trend (ranging market)
    - 20-40: Moderate trend
    - 40-60: Strong trend
    - 60+: Very strong trend

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices

    Returns:
        ADX value (0-100)
    """
    if len(closes) < ADX_PERIOD * 2 + 1:
        return 25.0  # Neutral when not enough data

    n = len(closes)

    # Calculate directional movement
    high_diff = np.diff(highs)
    low_diff = -np.diff(lows)

    # +DM and -DM
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # True Range for DI calculation
    prev_closes = closes[:-1]
    curr_highs = highs[1:]
    curr_lows = lows[1:]

    tr1 = curr_highs - curr_lows
    tr2 = np.abs(curr_highs - prev_closes)
    tr3 = np.abs(curr_lows - prev_closes)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Smooth with EMA
    smoothed_tr = ema(true_range, ADX_PERIOD)
    smoothed_plus_dm = ema(plus_dm, ADX_PERIOD)
    smoothed_minus_dm = ema(minus_dm, ADX_PERIOD)

    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / (smoothed_tr + 1e-10)
    minus_di = 100 * smoothed_minus_dm / (smoothed_tr + 1e-10)

    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * di_diff / (di_sum + 1e-10)

    # ADX is EMA of DX
    adx = ema(dx, ADX_PERIOD)

    return float(np.clip(adx[-1], 0, 100))


def compute_volatility_regime(closes: np.ndarray) -> float:
    """
    Compute volatility regime (percentile of current volatility).

    Returns value 0-1 indicating where current volatility sits
    relative to recent history:
    - 0.0: Very low volatility (bottom of range)
    - 0.5: Normal volatility
    - 1.0: Very high volatility (top of range)

    Args:
        closes: Array of closing prices

    Returns:
        Volatility percentile (0-1)
    """
    if len(closes) < VOLATILITY_REGIME_LOOKBACK:
        return 0.5

    # Calculate rolling volatility for each point
    returns = np.diff(closes) / closes[:-1]

    # Calculate volatility at each point using rolling window
    vol_window = min(20, len(returns) // 5)
    if vol_window < 5:
        return 0.5

    volatilities = []
    for i in range(vol_window, len(returns) + 1):
        vol = np.std(returns[i - vol_window:i])
        volatilities.append(vol)

    volatilities = np.array(volatilities)

    # Current volatility percentile
    current_vol = volatilities[-1]
    percentile = np.sum(volatilities <= current_vol) / len(volatilities)

    return float(percentile)


def compute_price_acceleration(closes: np.ndarray) -> float:
    """
    Compute price acceleration (2nd derivative / rate of change of momentum).

    Positive: Momentum is increasing (accelerating up or decelerating down)
    Negative: Momentum is decreasing (decelerating up or accelerating down)

    Args:
        closes: Array of closing prices

    Returns:
        Acceleration value (normalized, typically -0.01 to 0.01)
    """
    if len(closes) < MOMENTUM_PERIOD * 2:
        return 0.0

    # First derivative: momentum at each point
    # Use simple returns
    returns = np.diff(closes) / closes[:-1]

    # Second derivative: change in returns
    acceleration = np.diff(returns)

    # Average recent acceleration
    recent_acc = np.mean(acceleration[-MOMENTUM_PERIOD:])

    return float(recent_acc)


def compute_range_position(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """
    Compute position within the recent price range (-1 to 1).

    Shows where current price sits relative to recent high/low:
    - -1: At recent low (support area)
    - 0: At middle of range
    - +1: At recent high (resistance area)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices

    Returns:
        Range position (-1 to 1)
    """
    lookback = min(50, len(closes))

    if lookback < 5:
        return 0.0

    recent_high = np.max(highs[-lookback:])
    recent_low = np.min(lows[-lookback:])
    current = closes[-1]

    range_width = recent_high - recent_low
    if range_width == 0:
        return 0.0

    # Normalize to -1 to 1
    position = (current - recent_low) / range_width * 2 - 1

    return float(np.clip(position, -1, 1))


# ============================================
# MAIN FUNCTION - Compute all features at once
# ============================================

def compute_all_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray = None,
    lows: np.ndarray = None,
    include_regime: bool = False,
) -> dict:
    """
    Compute all technical indicators from price and volume arrays.

    This is the main function used for both real-time inference
    and training data generation.

    Args:
        closes: Array of closing prices (need at least MIN_CANDLES)
        volumes: Array of volumes (same length as closes)
        highs: Array of high prices (optional, for regime features)
        lows: Array of low prices (optional, for regime features)
        include_regime: If True, include ADX, ATR, and regime features

    Returns:
        Dictionary with all computed features

    Raises:
        ValueError: If not enough candles provided
    """
    # For backward compatibility, use MIN_CANDLES_BASIC if not using regime features
    min_required = MIN_CANDLES if include_regime else 35

    if len(closes) < min_required:
        raise ValueError(f"Need at least {min_required} candles, got {len(closes)}")

    if len(closes) != len(volumes):
        raise ValueError(f"closes and volumes must have same length")

    # Compute MACD (returns 3 values)
    macd, macd_signal, macd_histogram = compute_macd(closes)

    features = {
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

    # Add regime features if requested and data available
    if include_regime and highs is not None and lows is not None:
        if len(highs) != len(closes) or len(lows) != len(closes):
            raise ValueError("highs and lows must have same length as closes")

        features.update({
            "adx": compute_adx(highs, lows, closes),
            "atr": compute_atr(highs, lows, closes),
            "volatility_regime": compute_volatility_regime(closes),
            "price_acceleration": compute_price_acceleration(closes),
            "range_position": compute_range_position(highs, lows, closes),
        })

    return features


def normalize_features(features: dict, include_regime: bool = False) -> np.ndarray:
    """
    Convert feature dict to normalized numpy array for ML model.

    This applies the same normalization as FeatureVector.to_array()

    Args:
        features: Dictionary from compute_all_features()
        include_regime: If True, include regime features in output

    Returns:
        Numpy array of normalized features (9 base + 5 regime = 14 total)
    """
    base_features = [
        features["rsi"] / 100,              # Normalize to 0-1
        features["macd"],                    # Already normalized by price
        features["macd_signal"],
        features["macd_histogram"],
        features["ema_ratio"] - 1,           # Center around 0
        features["volatility"],
        features["volume_spike"] - 1,        # Center around 0
        features["momentum"],
        features["bollinger_position"],
    ]

    if include_regime and "adx" in features:
        regime_features = [
            features["adx"] / 100,           # Normalize to 0-1
            features["atr"] / features["price"],  # Normalize by price
            features["volatility_regime"],   # Already 0-1
            features["price_acceleration"] * 100,  # Scale up small values
            features["range_position"],      # Already -1 to 1
        ]
        base_features.extend(regime_features)

    return np.array(base_features)


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

REGIME_FEATURE_NAMES = [
    "adx",
    "atr",
    "volatility_regime",
    "price_acceleration",
    "range_position",
]

ALL_FEATURE_NAMES = FEATURE_NAMES + REGIME_FEATURE_NAMES
