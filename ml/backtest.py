"""
Backtesting Engine for VAPM Trading Agent.

Simulates trading on the test set using trained models with realistic
transaction costs, position sizing, and exit rules.

Usage:
    python backtest.py
    python backtest.py --model models/model_bundle_latest.joblib
    python backtest.py --compare  # Run backtest for all models from comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

MODELS_DIR = "models"

BASE_FEATURE_COLUMNS = [
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

ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "adx",
    "atr",
    "volatility_regime",
    "price_acceleration",
    "range_position",
]

TRAIN_RATIO = 0.8

# Trading parameters
INITIAL_CAPITAL = 10_000.0
POSITION_SIZE_PCT = 0.03       # 3% of capital per trade
ENTRY_CONFIDENCE = 0.60        # Min confidence to enter
REVERSAL_CONFIDENCE = 0.55     # Min confidence for reversal exit
STOP_LOSS_PCT = 0.02           # -2% stop loss
TAKE_PROFIT_PCT = 0.04         # +4% take profit
MAX_HOLD_CANDLES = 30          # Max 30 candles (30 minutes)

# Transaction costs
SLIPPAGE_BPS = 5               # 0.05% slippage per side
ROUND_TRIP_FEE_BPS = 10        # 0.10% round-trip fee

# Annualization factor for 1-min candles
MINUTES_PER_YEAR = 525_600
ANNUALIZATION_FACTOR = math.sqrt(MINUTES_PER_YEAR)


@dataclass(frozen=True)
class Trade:
    """Immutable record of a completed trade."""

    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    direction: str
    exit_reason: str


@dataclass
class Position:
    """Mutable position state during simulation."""

    entry_index: int
    entry_price: float
    direction: str
    size_pct: float


def load_model_bundle(path: str) -> dict:
    """Load a model bundle from disk."""
    bundle = joblib.load(path)
    return bundle


def compute_transaction_cost() -> float:
    """Compute total transaction cost as a fraction.

    Entry slippage + exit slippage + round-trip fee.
    """
    entry_slippage = SLIPPAGE_BPS / 10_000
    exit_slippage = SLIPPAGE_BPS / 10_000
    round_trip_fee = ROUND_TRIP_FEE_BPS / 10_000
    return entry_slippage + exit_slippage + round_trip_fee


def get_feature_columns(bundle: dict) -> list[str]:
    """Extract feature column names from a model bundle's metadata.

    Falls back to ALL_FEATURE_COLUMNS if metadata is not available.
    """
    metadata = bundle.get("metadata", {})
    features = metadata.get("features")
    if features:
        return features
    return ALL_FEATURE_COLUMNS


def run_backtest(
    model: Any,
    X_test: np.ndarray,
    prices: np.ndarray,
    scaler: Any | None = None,
) -> dict:
    """Run the backtest simulation on test data.

    Args:
        model: Trained sklearn-compatible classifier.
        X_test: Feature matrix for test period.
        prices: Price array aligned with X_test.
        scaler: Optional scaler for models that need it.

    Returns:
        Dictionary with backtest results and trade list.
    """
    n = len(X_test)
    transaction_cost = compute_transaction_cost()

    # Get predictions and probabilities
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    trades: list[Trade] = []
    position: Position | None = None
    capital = INITIAL_CAPITAL
    cumulative_pnl_curve: list[dict] = []

    for i in range(n):
        price = prices[i]
        pred = predictions[i]
        confidence = probabilities[i] if pred == 1 else (1.0 - probabilities[i])

        # Check exit conditions if we have a position
        if position is not None:
            hold_time = i - position.entry_index
            price_change_pct = (price - position.entry_price) / position.entry_price

            exit_reason = None

            # Stop loss
            if price_change_pct <= -STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            # Take profit
            elif price_change_pct >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            # Max hold time
            elif hold_time >= MAX_HOLD_CANDLES:
                exit_reason = "max_hold_time"
            # Reversal signal (prediction = DOWN with sufficient confidence)
            elif pred == 0 and (1.0 - probabilities[i]) >= REVERSAL_CONFIDENCE:
                exit_reason = "reversal_signal"

            if exit_reason is not None:
                # Close position
                raw_pnl_pct = price_change_pct
                net_pnl_pct = raw_pnl_pct - transaction_cost
                position_pnl = net_pnl_pct * position.size_pct * capital

                capital = capital + position_pnl

                trade = Trade(
                    entry_index=position.entry_index,
                    exit_index=i,
                    entry_price=position.entry_price,
                    exit_price=price,
                    pnl_pct=net_pnl_pct * 100,
                    direction=position.direction,
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                position = None

        # Check entry conditions if no position
        if position is None and pred == 1 and confidence >= ENTRY_CONFIDENCE:
            position = Position(
                entry_index=i,
                entry_price=price,
                direction="LONG",
                size_pct=POSITION_SIZE_PCT,
            )

        # Record cumulative PnL curve (sample every 10 candles to keep size manageable)
        if i % 10 == 0 or i == n - 1:
            cumulative_pnl_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            cumulative_pnl_curve.append({
                "candle_index": i,
                "cumulative_pnl_pct": round(cumulative_pnl_pct, 4),
            })

    # Force-close any remaining position at last price
    if position is not None:
        price = prices[-1]
        price_change_pct = (price - position.entry_price) / position.entry_price
        net_pnl_pct = price_change_pct - transaction_cost
        position_pnl = net_pnl_pct * position.size_pct * capital
        capital = capital + position_pnl

        trade = Trade(
            entry_index=position.entry_index,
            exit_index=n - 1,
            entry_price=position.entry_price,
            exit_price=price,
            pnl_pct=net_pnl_pct * 100,
            direction=position.direction,
            exit_reason="end_of_data",
        )
        trades.append(trade)

    # Compute summary metrics
    return compute_metrics(trades, capital, cumulative_pnl_curve, n)


def compute_metrics(
    trades: list[Trade],
    final_capital: float,
    cumulative_pnl_curve: list[dict],
    n_candles: int,
) -> dict:
    """Compute performance metrics from trade list."""
    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "cumulative_pnl_curve": cumulative_pnl_curve,
            "trades": [],
        }

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_win_pct = float(np.mean(wins)) if wins else 0.0
    avg_loss_pct = float(np.mean(losses)) if losses else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualized from per-trade returns)
    if len(pnls) > 1:
        mean_return = float(np.mean(pnls))
        std_return = float(np.std(pnls, ddof=1))
        if std_return > 0:
            # Estimate trades per year based on trade frequency
            avg_hold = float(np.mean([t.exit_index - t.entry_index for t in trades]))
            trades_per_year = MINUTES_PER_YEAR / max(avg_hold, 1)
            sharpe_ratio = (mean_return / std_return) * math.sqrt(trades_per_year)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown from cumulative PnL curve
    max_drawdown_pct = _compute_max_drawdown(cumulative_pnl_curve)

    # Exit reason distribution
    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    trade_records = [
        {
            "entry_index": t.entry_index,
            "exit_index": t.exit_index,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "direction": t.direction,
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]

    return {
        "total_return_pct": round(total_return_pct, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "total_trades": total_trades,
        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "exit_reasons": exit_reasons,
        "cumulative_pnl_curve": cumulative_pnl_curve,
        "trades": trade_records,
    }


def _compute_max_drawdown(curve: list[dict]) -> float:
    """Compute max drawdown percentage from a cumulative PnL curve."""
    if not curve:
        return 0.0

    peak = curve[0]["cumulative_pnl_pct"]
    max_dd = 0.0

    for point in curve:
        value = point["cumulative_pnl_pct"]
        if value > peak:
            peak = value
        dd = peak - value
        if dd > max_dd:
            max_dd = dd

    return max_dd


def run_comparison_backtest(
    df_test: pd.DataFrame,
    prices: np.ndarray,
    comparison_path: str,
) -> dict:
    """Run backtest for all models from the comparison JSON.

    Args:
        df_test: Test DataFrame with all feature columns available.
        prices: Price array aligned with df_test.
        comparison_path: Path to model_comparison.json.
    """
    with open(comparison_path) as f:
        comparison = json.load(f)

    per_model_results = {}

    for model_info in comparison["models"]:
        model_name = model_info["model_name"]
        model_path = model_info["model_path"]

        if not os.path.exists(model_path):
            logger.warning("Model file not found: %s, skipping", model_path)
            continue

        logger.info("Running backtest for %s (%s)...", model_name, model_path)

        bundle = load_model_bundle(model_path)
        model = bundle["model"]
        scaler = bundle.get("scaler")
        feature_cols = get_feature_columns(bundle)
        X_test = df_test[feature_cols].values

        result = run_backtest(model, X_test, prices, scaler=scaler)

        # Store summary (exclude full trade list and curve for per_model to keep JSON reasonable)
        per_model_results[model_name] = {
            "total_return_pct": result["total_return_pct"],
            "win_rate": result["win_rate"],
            "profit_factor": result["profit_factor"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown_pct": result["max_drawdown_pct"],
            "total_trades": result["total_trades"],
            "avg_win_pct": result["avg_win_pct"],
            "avg_loss_pct": result["avg_loss_pct"],
            "exit_reasons": result.get("exit_reasons", {}),
        }

        logger.info(
            "%s: Return=%.4f%%, WinRate=%.4f, Sharpe=%.4f, Trades=%d",
            model_name,
            result["total_return_pct"],
            result["win_rate"],
            result["sharpe_ratio"],
            result["total_trades"],
        )

    return per_model_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest VAPM trading strategy")
    parser.add_argument(
        "--model",
        type=str,
        default="models/model_bundle_latest.joblib",
        help="Path to model bundle",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/training_data_v2.csv",
        help="Input training data CSV",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run backtest for all models from model_comparison.json",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BACKTESTING ENGINE")
    logger.info("=" * 60)

    # Load data
    df = pd.read_csv(args.input)
    logger.info("Loaded %d samples from %s", len(df), args.input)

    prices = df["price"].values

    # Time-based test split (same as training)
    split_idx = int(len(df) * TRAIN_RATIO)
    prices_test = prices[split_idx:]

    # Run primary backtest with latest model
    logger.info("Loading primary model: %s", args.model)
    bundle = load_model_bundle(args.model)
    model = bundle["model"]
    scaler = bundle.get("scaler")
    feature_cols = get_feature_columns(bundle)
    logger.info("Model features (%d): %s", len(feature_cols), feature_cols)

    X_test = df[feature_cols].values[split_idx:]
    logger.info("Test set: %d samples (last %.0f%%)", len(X_test), (1 - TRAIN_RATIO) * 100)
    logger.info(
        "Price range: $%.2f - $%.2f",
        prices_test.min(),
        prices_test.max(),
    )

    logger.info("Running primary backtest...")
    results = run_backtest(model, X_test, prices_test, scaler=scaler)

    # Run comparison if requested and comparison file exists
    comparison_path = os.path.join(MODELS_DIR, "model_comparison.json")
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    if args.compare and os.path.exists(comparison_path):
        logger.info("Running comparison backtests...")
        per_model = run_comparison_backtest(df_test, prices_test, comparison_path)
        results["per_model"] = per_model
    elif args.compare:
        logger.warning(
            "model_comparison.json not found at %s. "
            "Run train_model_comparison.py first.",
            comparison_path,
        )

    # Add metadata
    results["metadata"] = {
        "model_path": args.model,
        "dataset": args.input,
        "test_samples": len(X_test),
        "initial_capital": INITIAL_CAPITAL,
        "position_size_pct": POSITION_SIZE_PCT,
        "entry_confidence": ENTRY_CONFIDENCE,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "max_hold_candles": MAX_HOLD_CANDLES,
        "slippage_bps": SLIPPAGE_BPS,
        "round_trip_fee_bps": ROUND_TRIP_FEE_BPS,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    # Summary
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("Total Return:   %.4f%%", results["total_return_pct"])
    logger.info("Win Rate:       %.4f", results["win_rate"])
    logger.info("Profit Factor:  %.4f", results["profit_factor"])
    logger.info("Sharpe Ratio:   %.4f", results["sharpe_ratio"])
    logger.info("Max Drawdown:   %.4f%%", results["max_drawdown_pct"])
    logger.info("Total Trades:   %d", results["total_trades"])
    logger.info("Avg Win:        %.4f%%", results["avg_win_pct"])
    logger.info("Avg Loss:       %.4f%%", results["avg_loss_pct"])

    if "exit_reasons" in results:
        logger.info("Exit Reasons:   %s", results["exit_reasons"])

    # Save results
    output_path = os.path.join(os.path.dirname(args.input), "..", "backtest_results.json")
    output_path = os.path.normpath(output_path)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
