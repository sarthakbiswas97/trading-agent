"""
Prediction Horizon Experiment

Tests different prediction horizons (1, 3, 5, 10, 15 min) to find
the optimal balance between signal clarity and noise.

Usage:
    cd ml/experiments
    python horizon_test.py --input ../candles_3months.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add backend to path
sys.path.insert(0, '../../backend')
from core.indicators import compute_all_features, MIN_CANDLES, FEATURE_NAMES


# ============================================
# CONFIGURATION
# ============================================

HORIZONS = [1, 3, 5, 10, 15]  # Prediction horizons to test (in candles/minutes)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def prepare_data_for_horizon(df: pd.DataFrame, horizon: int) -> tuple:
    """
    Prepare features and labels for a specific prediction horizon.

    Args:
        df: DataFrame with candle data
        horizon: Number of candles to look ahead

    Returns:
        X, y: Features and labels
    """
    features_list = []
    targets = []

    start_idx = MIN_CANDLES - 1  # First valid index
    end_idx = len(df) - horizon   # Last valid index (need horizon candles ahead)

    for i in range(start_idx, end_idx):
        # Get window of candles
        window_start = i - MIN_CANDLES + 1
        window_end = i + 1

        window = df.iloc[window_start:window_end]
        closes = window['close'].values
        volumes = window['volume'].values

        # Compute features
        features = compute_all_features(closes, volumes)

        # Normalize features for XGBoost (same as in data_preparation.py)
        feature_array = [
            features['rsi'] / 100,
            features['macd'],
            features['macd_signal'],
            features['macd_histogram'],
            features['ema_ratio'] - 1,
            features['volatility'],
            features['volume_spike'] - 1,
            features['momentum'],
            features['bollinger_position'],
        ]
        features_list.append(feature_array)

        # Target: price UP after `horizon` candles
        current_price = df.iloc[i]['close']
        future_price = df.iloc[i + horizon]['close']
        targets.append(1 if future_price > current_price else 0)

    X = np.array(features_list)
    y = np.array(targets)

    return X, y


def train_and_evaluate(X: np.ndarray, y: np.ndarray, horizon: int) -> dict:
    """
    Train XGBoost and evaluate performance.

    Returns dict with metrics.
    """
    # Time-based split (no shuffle to prevent look-ahead bias)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Baseline (always predict majority class)
    majority_class = 1 if y_train.sum() > len(y_train) / 2 else 0
    baseline_accuracy = (y_test == majority_class).mean()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Class distribution
    train_up_pct = y_train.mean() * 100
    test_up_pct = y_test.mean() * 100

    return {
        "horizon": horizon,
        "samples": len(y),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_up_pct": train_up_pct,
        "test_up_pct": test_up_pct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "baseline": baseline_accuracy,
        "improvement": accuracy - baseline_accuracy,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def main():
    parser = argparse.ArgumentParser(description="Test prediction horizons")
    parser.add_argument("--input", type=str, default="../candles_3months.csv", help="Input candles CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("PREDICTION HORIZON EXPERIMENT")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Horizons to test: {HORIZONS} (minutes/candles)")
    print(f"Test size: {TEST_SIZE * 100:.0f}%")
    print("=" * 70)

    # Load data
    print("\nLoading candle data...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} candles")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Test each horizon
    results = []

    for horizon in HORIZONS:
        print(f"\n" + "-" * 70)
        print(f"Testing horizon = {horizon} minutes")
        print("-" * 70)

        X, y = prepare_data_for_horizon(df, horizon)
        print(f"  Samples: {len(y):,}")
        print(f"  UP class: {y.mean() * 100:.1f}%")

        metrics = train_and_evaluate(X, y, horizon)
        results.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f} (baseline: {metrics['baseline']:.4f})")
        print(f"  Improvement: {metrics['improvement']:+.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Create DataFrame for display
    results_df = pd.DataFrame(results)
    results_df = results_df[['horizon', 'accuracy', 'baseline', 'improvement', 'precision', 'recall', 'f1']]
    results_df.columns = ['Horizon (min)', 'Accuracy', 'Baseline', 'Improvement', 'Precision', 'Recall', 'F1']

    print(results_df.to_string(index=False, float_format='%.4f'))

    # Find best horizon
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    best_by_accuracy = max(results, key=lambda x: x['accuracy'])
    best_by_improvement = max(results, key=lambda x: x['improvement'])
    best_by_f1 = max(results, key=lambda x: x['f1'])

    print(f"Best by Accuracy: {best_by_accuracy['horizon']} min ({best_by_accuracy['accuracy']:.4f})")
    print(f"Best by Improvement: {best_by_improvement['horizon']} min ({best_by_improvement['improvement']:+.4f})")
    print(f"Best by F1 Score: {best_by_f1['horizon']} min ({best_by_f1['f1']:.4f})")

    # Recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)

    # Score each horizon (weighted combination)
    for r in results:
        # Normalize metrics to 0-1 scale for fair comparison
        r['score'] = (
            r['improvement'] * 0.4 +  # Weight improvement heavily (shows true predictive power)
            r['f1'] * 0.3 +           # F1 balances precision/recall
            r['recall'] * 0.3          # Important to catch UP moves
        )

    best_overall = max(results, key=lambda x: x['score'])

    print(f"Recommended horizon: {best_overall['horizon']} minutes")
    print(f"  - Accuracy: {best_overall['accuracy']:.4f} ({best_overall['improvement']:+.4f} vs baseline)")
    print(f"  - Precision: {best_overall['precision']:.4f}")
    print(f"  - Recall: {best_overall['recall']:.4f}")
    print(f"  - F1: {best_overall['f1']:.4f}")

    # Save results
    output_file = f"horizon_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
