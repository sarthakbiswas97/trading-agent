"""
XGBoost Model Training v3 - With Regime Features

Builds on v2 (GridSearchCV) and adds:
- Regime features (ADX, ATR, volatility_regime, price_acceleration, range_position)
- Comparison between base and regime-enhanced models

Usage:
    python train_model_v3.py --input dataset/training_data_v2.csv
    python train_model_v3.py --input dataset/training_data_v2.csv --quick
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# ============================================
# CONFIGURATION
# ============================================

MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = 'models'

# Base features (original 9)
BASE_FEATURE_COLUMNS = [
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

# Regime features (new 5)
REGIME_FEATURE_COLUMNS = [
    'adx',
    'atr',
    'volatility_regime',
    'price_acceleration',
    'range_position',
]

# All features
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS

# Split ratios
TRAIN_RATIO = 0.8

# Parameter grid for GridSearchCV
PARAM_GRID_QUICK = {
    'max_depth': [4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
}

PARAM_GRID_FULL = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 150, 200],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 0.9],
}

# Base XGBoost parameters
XGB_BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}


def load_data(input_file: str) -> tuple:
    """Load training data."""
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} samples from {input_file}")
    print(f"Columns: {list(df.columns)}")

    # Check for regime features
    has_regime = all(col in df.columns for col in REGIME_FEATURE_COLUMNS)
    print(f"Regime features present: {has_regime}")

    return df, has_regime


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    model_name: str,
    quick_mode: bool = True,
) -> dict:
    """Train with GridSearchCV and evaluate."""

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Features: {len(feature_names)}")
    print(f"{'='*60}")

    # GridSearchCV
    param_grid = PARAM_GRID_QUICK if quick_mode else PARAM_GRID_FULL
    n_splits = 3 if quick_mode else 5

    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_model = XGBClassifier(**XGB_BASE_PARAMS)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print(f"Running GridSearchCV ({len(param_grid)} params, {n_splits} folds)...")
    grid_search.fit(X_train, y_train)

    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    print(f"Best params: {grid_search.best_params_}")

    # Train final model with early stopping
    final_params = {**XGB_BASE_PARAMS, **grid_search.best_params_}
    final_params['early_stopping_rounds'] = 15

    val_size = int(len(X_train) * 0.1)
    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    model = XGBClassifier(**final_params)
    model.fit(X_train_fit, y_train_fit, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    baseline = max(y_test.mean(), 1 - y_test.mean())
    improvement = accuracy - baseline

    print(f"\n=== Test Results: {model_name} ===")
    print(f"Accuracy:    {accuracy:.4f} (baseline: {baseline:.4f}, {improvement:+.4f})")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC AUC:     {auc:.4f}")

    # Feature importance
    importance = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nTop 5 Features:")
    for feat, imp in feature_importance[:5]:
        print(f"  {feat:25s}: {imp:.4f}")

    return {
        'model_name': model_name,
        'model': model,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'baseline': baseline,
        'improvement': improvement,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with regime features")
    parser.add_argument("--input", type=str, default="dataset/training_data_v2.csv", help="Input CSV")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()

    print("=" * 70)
    print("XGBOOST MODEL TRAINING v3 - With Regime Features")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Model version: {MODEL_VERSION}")
    print("=" * 70)

    # Load data
    df, has_regime = load_data(args.input)

    if not has_regime:
        print("ERROR: Dataset does not contain regime features!")
        print("Please run data_preparation_v2.py first.")
        return

    # Split data
    y = df['target'].values
    split_idx = int(len(df) * TRAIN_RATIO)

    results = []

    # ===== Model 1: Base features only =====
    X_base = df[BASE_FEATURE_COLUMNS].values
    X_base_train, X_base_test = X_base[:split_idx], X_base[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    result_base = train_and_evaluate(
        X_base_train, y_train, X_base_test, y_test,
        BASE_FEATURE_COLUMNS, "Base Features (9)",
        quick_mode=args.quick
    )
    results.append(result_base)

    # ===== Model 2: All features (base + regime) =====
    X_all = df[ALL_FEATURE_COLUMNS].values
    X_all_train, X_all_test = X_all[:split_idx], X_all[split_idx:]

    result_all = train_and_evaluate(
        X_all_train, y_train, X_all_test, y_test,
        ALL_FEATURE_COLUMNS, "All Features (14)",
        quick_mode=args.quick
    )
    results.append(result_all)

    # ===== Comparison =====
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Accuracy': r['accuracy'],
        'Improvement': r['improvement'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1': r['f1'],
        'AUC': r['auc'],
    } for r in results])

    print(comparison_df.to_string(index=False, float_format='%.4f'))

    # Calculate improvement from adding regime features
    base_acc = result_base['accuracy']
    all_acc = result_all['accuracy']
    regime_gain = all_acc - base_acc

    print(f"\nRegime Feature Impact:")
    print(f"  Base accuracy:        {base_acc:.4f}")
    print(f"  With regime features: {all_acc:.4f}")
    print(f"  Regime feature gain:  {regime_gain:+.4f}")

    # Save best model
    best_result = max(results, key=lambda r: r['f1'])
    print(f"\nBest model by F1: {best_result['model_name']}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_output = f'{MODELS_DIR}/model_bundle_v{MODEL_VERSION}.joblib'
    latest_model = f'{MODELS_DIR}/model_bundle_latest.joblib'

    metadata = {
        'version': MODEL_VERSION,
        'timestamp': datetime.now().isoformat(),
        'features': best_result['feature_names'],
        'feature_count': len(best_result['feature_names']),
        'includes_regime_features': 'adx' in best_result['feature_names'],
        'hyperparameters': best_result['best_params'],
        'results': {
            'accuracy': best_result['accuracy'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'f1': best_result['f1'],
            'auc': best_result['auc'],
            'improvement': best_result['improvement'],
        },
        'feature_importance': {f: float(i) for f, i in best_result['feature_importance']},
        'comparison': {
            'base_accuracy': result_base['accuracy'],
            'all_accuracy': result_all['accuracy'],
            'regime_gain': regime_gain,
        },
    }

    model_bundle = {
        'model': best_result['model'],
        'metadata': metadata,
    }

    joblib.dump(model_bundle, model_output)
    joblib.dump(model_bundle, latest_model)

    metrics_file = f'{MODELS_DIR}/metrics_v{MODEL_VERSION}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model: {model_output}")
    print(f"Latest: {latest_model}")
    print(f"Metrics: {metrics_file}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
Model Version: {MODEL_VERSION}

Best Model: {best_result['model_name']}
  Accuracy:    {best_result['accuracy']*100:.2f}% ({best_result['improvement']*100:+.2f}% vs baseline)
  Precision:   {best_result['precision']:.4f}
  Recall:      {best_result['recall']:.4f}
  F1 Score:    {best_result['f1']:.4f}
  ROC AUC:     {best_result['auc']:.4f}

Regime Feature Impact: {regime_gain*100:+.2f}% accuracy gain

Top 3 Features:
  1. {best_result['feature_importance'][0][0]}: {best_result['feature_importance'][0][1]:.4f}
  2. {best_result['feature_importance'][1][0]}: {best_result['feature_importance'][1][1]:.4f}
  3. {best_result['feature_importance'][2][0]}: {best_result['feature_importance'][2][1]:.4f}
""")


if __name__ == "__main__":
    main()
