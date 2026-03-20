"""
Walk-Forward Validation

Tests model robustness across different market regimes by:
1. Training on a rolling window of past data
2. Testing on the next period (out-of-sample)
3. Rolling forward and repeating
4. Aggregating metrics across all folds

This approach:
- Prevents look-ahead bias
- Tests model on unseen market conditions
- Reveals how stable the model is across regimes

Usage:
    cd ml/experiments
    python walk_forward_validation.py --input ../dataset/training_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Add backend to path
sys.path.insert(0, '../../backend')


# ============================================
# CONFIGURATION
# ============================================

# Features (must match training)
FEATURE_COLUMNS = [
    'rsi',
    'macd',
    'macd_signal',
    'macd_histogram',
    'ema_ratio',
    'volatility',
    'volume_spike',
    'momentum',
    'bollinger_position',
]

# Walk-forward parameters
TRAIN_WINDOW_DAYS = 60      # Train on 60 days (2 months)
TEST_WINDOW_DAYS = 14       # Test on 14 days (2 weeks)
STEP_DAYS = 14              # Roll forward by 14 days

CANDLES_PER_DAY = 24 * 60   # 1-minute candles

# Best params from GridSearchCV
BEST_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 150,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}


def run_walk_forward(df: pd.DataFrame) -> list[dict]:
    """
    Run walk-forward validation.

    Returns list of fold results.
    """
    X = df[FEATURE_COLUMNS].values
    y = df['target'].values

    # Convert to candle counts
    train_size = TRAIN_WINDOW_DAYS * CANDLES_PER_DAY
    test_size = TEST_WINDOW_DAYS * CANDLES_PER_DAY
    step_size = STEP_DAYS * CANDLES_PER_DAY

    print(f"\nWalk-Forward Configuration:")
    print(f"  Train window: {TRAIN_WINDOW_DAYS} days ({train_size:,} candles)")
    print(f"  Test window: {TEST_WINDOW_DAYS} days ({test_size:,} candles)")
    print(f"  Step size: {STEP_DAYS} days ({step_size:,} candles)")
    print(f"  Total data: {len(df):,} candles")

    # Calculate number of folds
    max_start = len(df) - train_size - test_size
    n_folds = (max_start // step_size) + 1

    print(f"  Expected folds: {n_folds}")

    results = []
    fold_num = 0
    start_idx = 0

    while start_idx + train_size + test_size <= len(df):
        fold_num += 1

        # Define train/test ranges
        train_end = start_idx + train_size
        test_end = train_end + test_size

        X_train = X[start_idx:train_end]
        y_train = y[start_idx:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        print(f"\n{'='*60}")
        print(f"Fold {fold_num}")
        print(f"{'='*60}")
        print(f"  Train: indices {start_idx:,} to {train_end:,} ({len(X_train):,} samples)")
        print(f"  Test:  indices {train_end:,} to {test_end:,} ({len(X_test):,} samples)")

        # Train model
        model = XGBClassifier(**BEST_PARAMS)

        # Split train for early stopping
        val_size = int(len(X_train) * 0.1)
        X_train_fit = X_train[:-val_size]
        y_train_fit = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        # Baseline for this fold
        baseline = max(y_test.mean(), 1 - y_test.mean())
        improvement = accuracy - baseline

        # Class distribution in test set
        test_up_pct = y_test.mean() * 100

        fold_result = {
            'fold': fold_num,
            'train_start': start_idx,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_up_pct': test_up_pct,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'baseline': baseline,
            'improvement': improvement,
        }

        results.append(fold_result)

        print(f"\n  Results:")
        print(f"    Accuracy:    {accuracy:.4f} (baseline: {baseline:.4f}, {improvement:+.4f})")
        print(f"    Precision:   {precision:.4f}")
        print(f"    Recall:      {recall:.4f}")
        print(f"    F1:          {f1:.4f}")
        print(f"    ROC AUC:     {auc:.4f}")
        print(f"    Test UP %:   {test_up_pct:.1f}%")

        # Roll forward
        start_idx += step_size

    return results


def analyze_results(results: list[dict]) -> dict:
    """Analyze walk-forward results."""

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results)

    # Summary statistics
    print("\n=== Summary Statistics ===")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'improvement']

    summary = {}
    for metric in metrics:
        values = df_results[metric].values
        summary[metric] = {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
        }
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {values.mean():.4f} (+/- {values.std():.4f})")
        print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

    # Per-fold table
    print("\n=== Per-Fold Results ===")
    print(df_results[['fold', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'improvement']].to_string(index=False))

    # Stability analysis
    print("\n=== Stability Analysis ===")

    # Count folds that beat baseline
    folds_above_baseline = (df_results['improvement'] > 0).sum()
    total_folds = len(df_results)

    print(f"Folds beating baseline: {folds_above_baseline}/{total_folds} ({folds_above_baseline/total_folds*100:.0f}%)")

    # Consistency (std/mean ratio - lower is better)
    accuracy_cv = df_results['accuracy'].std() / df_results['accuracy'].mean()
    print(f"Accuracy CV (std/mean): {accuracy_cv:.4f} (lower = more consistent)")

    # Check for degradation over time
    early_folds = df_results.head(len(df_results)//2)
    late_folds = df_results.tail(len(df_results)//2)

    early_acc = early_folds['accuracy'].mean()
    late_acc = late_folds['accuracy'].mean()

    print(f"\nTime Stability:")
    print(f"  Early folds avg accuracy: {early_acc:.4f}")
    print(f"  Late folds avg accuracy:  {late_acc:.4f}")
    print(f"  Trend: {'Stable' if abs(early_acc - late_acc) < 0.01 else ('Degrading' if late_acc < early_acc else 'Improving')}")

    # Recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)

    mean_improvement = df_results['improvement'].mean()
    mean_f1 = df_results['f1'].mean()

    if mean_improvement > 0.02 and folds_above_baseline >= total_folds * 0.8:
        verdict = "STRONG: Model consistently outperforms baseline"
    elif mean_improvement > 0.01 and folds_above_baseline >= total_folds * 0.6:
        verdict = "MODERATE: Model shows some predictive power but inconsistent"
    elif mean_improvement > 0:
        verdict = "WEAK: Model barely outperforms baseline on average"
    else:
        verdict = "FAIL: Model does not outperform baseline"

    print(f"\nVerdict: {verdict}")
    print(f"Mean improvement over baseline: {mean_improvement:+.4f}")
    print(f"Mean F1 score: {mean_f1:.4f}")
    print(f"Consistency (folds above baseline): {folds_above_baseline}/{total_folds}")

    return {
        'summary': summary,
        'n_folds': total_folds,
        'folds_above_baseline': folds_above_baseline,
        'mean_improvement': mean_improvement,
        'mean_f1': mean_f1,
        'verdict': verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--input", type=str, default="../dataset/training_data.csv", help="Input CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("Testing model robustness across market regimes")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Best params (from GridSearchCV): {BEST_PARAMS}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} samples")

    # Run walk-forward
    results = run_walk_forward(df)

    # Analyze
    analysis = analyze_results(results)

    # Save results
    output_file = f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
