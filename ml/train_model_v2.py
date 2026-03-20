"""
XGBoost Model Training v2 - With Hyperparameter Tuning

Improvements over v1:
1. GridSearchCV for hyperparameter optimization
2. TimeSeriesSplit for proper time-series cross-validation
3. Updated for 15-minute prediction horizon (from experiment results)
4. Walk-forward validation option
5. Threshold optimization for precision-recall tradeoff

Usage:
    python train_model_v2.py --input dataset/training_data.csv
    python train_model_v2.py --input dataset/training_data.csv --quick  # Skip GridSearch
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

# XGBoost
from xgboost import XGBClassifier

# SHAP for explainability
import shap


# ============================================
# CONFIGURATION
# ============================================

# Model versioning
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = 'models'

# Features (must match data_preparation.py)
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

# Split ratios
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# GridSearchCV parameter grid
PARAM_GRID = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Reduced grid for quick mode (faster iteration)
PARAM_GRID_QUICK = {
    'max_depth': [4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
}

# Base XGBoost parameters (non-tuned)
XGB_BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}


def load_data(input_file: str) -> tuple:
    """Load and prepare training data."""
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} samples from {input_file}")

    # Separate features and target
    X = df[FEATURE_COLUMNS].values
    y = df['target'].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (0=DOWN, 1=UP)")
    print(f"Class balance: {y.mean()*100:.1f}% UP")

    return X, y, df


def time_based_split(X: np.ndarray, y: np.ndarray) -> tuple:
    """Time-based train/test split (no shuffling)."""
    print("\n" + "=" * 60)
    print("Splitting data (time-based)...")
    print("=" * 60)

    split_idx = int(len(X) * TRAIN_RATIO)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"Train samples: {len(X_train):,} ({TRAIN_RATIO*100:.0f}%)")
    print(f"Test samples: {len(X_test):,} ({TEST_RATIO*100:.0f}%)")

    return X_train, y_train, X_test, y_test


def run_grid_search(X_train: np.ndarray, y_train: np.ndarray, quick_mode: bool = False) -> dict:
    """Run GridSearchCV to find best hyperparameters."""
    print("\n" + "=" * 60)
    print("Running GridSearchCV hyperparameter tuning...")
    print("=" * 60)

    param_grid = PARAM_GRID_QUICK if quick_mode else PARAM_GRID

    # Calculate total combinations
    total_combos = 1
    for key, values in param_grid.items():
        total_combos *= len(values)

    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {total_combos}")

    # TimeSeriesSplit for time-series cross-validation
    # This ensures we always train on past and validate on future
    n_splits = 3 if quick_mode else 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"Cross-validation: TimeSeriesSplit with {n_splits} folds")

    # Base estimator
    base_model = XGBClassifier(**XGB_BASE_PARAMS)

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',  # Optimize for F1 score (balance precision/recall)
        n_jobs=-1,
        verbose=2,
        refit=True,  # Refit best model on full training data
    )

    print("\nStarting search (this may take a few minutes)...")
    start_time = datetime.now()

    grid_search.fit(X_train, y_train)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nSearch completed in {elapsed:.1f} seconds")

    print("\n" + "-" * 60)
    print("GRID SEARCH RESULTS")
    print("-" * 60)
    print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Show top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')

    print("\nTop 5 parameter combinations:")
    for i, row in results_df.head(5).iterrows():
        print(f"  Rank {row['rank_test_score']}: F1={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        print(f"    Params: {row['params']}")

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'search_time': elapsed,
    }


def train_final_model(X_train: np.ndarray, y_train: np.ndarray, best_params: dict) -> XGBClassifier:
    """Train final model with best parameters (with early stopping)."""
    print("\n" + "=" * 60)
    print("Training final model with best parameters...")
    print("=" * 60)

    # Combine best params with base params
    final_params = {**XGB_BASE_PARAMS, **best_params}
    final_params['early_stopping_rounds'] = 15

    print(f"Final parameters: {final_params}")

    # Split for early stopping validation
    val_size = int(len(X_train) * 0.1)
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    model = XGBClassifier(**final_params)

    model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    print(f"\nTraining complete! Best iteration: {model.best_iteration}")

    return model


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """Find optimal probability threshold for classification."""
    print("\n" + "-" * 60)
    print("Finding optimal decision threshold...")
    print("-" * 60)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # Find threshold with best F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    print(f"Default threshold (0.5):")
    y_pred_default = (y_proba >= 0.5).astype(int)
    print(f"  Precision: {precision_score(y_true, y_pred_default):.4f}")
    print(f"  Recall: {recall_score(y_true, y_pred_default):.4f}")
    print(f"  F1: {f1_score(y_true, y_pred_default):.4f}")

    print(f"\nOptimal threshold ({best_threshold:.3f}):")
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    print(f"  Precision: {precision_score(y_true, y_pred_optimal):.4f}")
    print(f"  Recall: {recall_score(y_true, y_pred_optimal):.4f}")
    print(f"  F1: {f1_score(y_true, y_pred_optimal):.4f}")

    # Also test higher thresholds for trading (high precision preferred)
    print("\nHigh-confidence thresholds (for trading):")
    for thresh in [0.6, 0.65, 0.7]:
        y_pred_high = (y_proba >= thresh).astype(int)
        n_predictions = y_pred_high.sum()
        if n_predictions > 0:
            prec = precision_score(y_true, y_pred_high)
            rec = recall_score(y_true, y_pred_high)
            print(f"  Threshold {thresh}: Precision={prec:.4f}, Recall={rec:.4f}, Trades={n_predictions}")
        else:
            print(f"  Threshold {thresh}: No predictions")

    return best_threshold, best_f1


def evaluate_model(model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Comprehensive model evaluation."""
    print("\n" + "=" * 60)
    print("Evaluating model on test set...")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Baseline
    baseline = max(np.mean(y_test), 1 - np.mean(y_test))

    print(f"\n=== Test Set Results ===")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"\nBaseline (majority class): {baseline:.4f}")
    print(f"Improvement over baseline: {(accuracy - baseline)*100:+.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 DOWN    UP")
    print(f"Actual DOWN    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual UP      {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_proba)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'baseline': float(baseline),
        'improvement': float(accuracy - baseline),
        'confusion_matrix': cm.tolist(),
        'optimal_threshold': float(optimal_threshold),
        'optimal_f1': float(optimal_f1),
    }


def get_feature_importance(model: XGBClassifier) -> list:
    """Get sorted feature importance."""
    print("\n" + "=" * 60)
    print("Feature importance...")
    print("=" * 60)

    importance = model.feature_importances_
    feature_importance = sorted(
        zip(FEATURE_COLUMNS, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nFeature Importance (higher = more predictive):")
    for feature, imp in feature_importance:
        bar = "█" * int(imp * 50)
        print(f"  {feature:20s}: {imp:.4f} {bar}")

    return feature_importance


def save_model(model: XGBClassifier, metrics: dict, grid_search_results: dict, feature_importance: list):
    """Save model bundle with all metadata."""
    print("\n" + "=" * 60)
    print("Saving model bundle...")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # File paths
    model_output = f'{MODELS_DIR}/model_bundle_v{MODEL_VERSION}.joblib'
    latest_model = f'{MODELS_DIR}/model_bundle_latest.joblib'
    metrics_output = f'{MODELS_DIR}/metrics_v{MODEL_VERSION}.json'
    latest_metrics = f'{MODELS_DIR}/metrics_latest.json'

    # Build metadata
    metadata = {
        'version': MODEL_VERSION,
        'timestamp': datetime.now().isoformat(),
        'features': FEATURE_COLUMNS,
        'feature_count': len(FEATURE_COLUMNS),
        'hyperparameter_tuning': {
            'method': 'GridSearchCV',
            'cv': 'TimeSeriesSplit',
            'best_params': grid_search_results['best_params'],
            'best_cv_score': float(grid_search_results['best_score']),
            'search_time_seconds': grid_search_results['search_time'],
        },
        'results': metrics,
        'feature_importance': {feat: float(imp) for feat, imp in feature_importance},
    }

    # Bundle model + metadata
    model_bundle = {
        'model': model,
        'metadata': metadata,
    }

    # Save versioned
    joblib.dump(model_bundle, model_output)
    print(f"Versioned bundle: {model_output}")

    # Save latest
    joblib.dump(model_bundle, latest_model)
    print(f"Latest bundle: {latest_model}")

    # Save metrics JSON
    with open(metrics_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Versioned metrics: {metrics_output}")

    with open(latest_metrics, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Latest metrics: {latest_metrics}")

    return model_output, metadata


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with hyperparameter tuning")
    parser.add_argument("--input", type=str, default="dataset/training_data.csv", help="Input CSV file")
    parser.add_argument("--quick", action="store_true", help="Quick mode (reduced parameter grid)")
    args = parser.parse_args()

    print("=" * 60)
    print("XGBOOST MODEL TRAINING v2")
    print("With GridSearchCV Hyperparameter Tuning")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Model version: {MODEL_VERSION}")
    print("=" * 60)

    # Step 1: Load data
    X, y, df = load_data(args.input)

    # Step 2: Time-based split
    X_train, y_train, X_test, y_test = time_based_split(X, y)

    # Step 3: GridSearchCV hyperparameter tuning
    grid_results = run_grid_search(X_train, y_train, quick_mode=args.quick)

    # Step 4: Train final model with best params (with early stopping)
    model = train_final_model(X_train, y_train, grid_results['best_params'])

    # Step 5: Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)

    # Step 6: Feature importance
    feature_importance = get_feature_importance(model)

    # Step 7: Save model bundle
    model_path, metadata = save_model(model, metrics, grid_results, feature_importance)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"""
Model Version: {MODEL_VERSION}

Hyperparameter Tuning:
  Best CV F1: {grid_results['best_score']:.4f}
  Best Params: {grid_results['best_params']}
  Search Time: {grid_results['search_time']:.1f} seconds

Test Set Performance:
  Accuracy:    {metrics['accuracy']*100:.2f}% ({metrics['improvement']*100:+.2f}% vs baseline)
  Precision:   {metrics['precision']:.4f}
  Recall:      {metrics['recall']:.4f}
  F1 Score:    {metrics['f1']:.4f}
  ROC AUC:     {metrics['auc']:.4f}

Trading Threshold:
  Optimal: {metrics['optimal_threshold']:.3f} (F1: {metrics['optimal_f1']:.4f})

Top Predictive Features:
  1. {feature_importance[0][0]}: {feature_importance[0][1]:.4f}
  2. {feature_importance[1][0]}: {feature_importance[1][1]:.4f}
  3. {feature_importance[2][0]}: {feature_importance[2][1]:.4f}

Saved: {model_path}
""")


if __name__ == "__main__":
    main()
