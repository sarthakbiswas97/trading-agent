"""
Model Comparison: XGBoost vs LightGBM vs Logistic Regression.

Trains three models on the same data split using TimeSeriesSplit CV,
then evaluates each on the held-out test set. Results are saved to
ml/models/model_comparison.json for use by the backtesting engine.

Usage:
    python train_model_comparison.py
    python train_model_comparison.py --input dataset/training_data_v2.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

MODELS_DIR = "models"

# Features matching the existing pipeline (base 9 used by latest model)
FEATURE_COLUMNS = [
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

TRAIN_RATIO = 0.8
RANDOM_STATE = 42
N_CV_SPLITS = 5


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a model to train."""

    name: str
    estimator: Any
    needs_scaling: bool


def build_model_specs() -> tuple[ModelSpec, ...]:
    """Create the three model specifications."""
    xgb_spec = ModelSpec(
        name="XGBoost",
        estimator=XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        needs_scaling=False,
    )

    lgbm_spec = ModelSpec(
        name="LightGBM",
        estimator=lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            objective="binary",
            metric="binary_logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        needs_scaling=False,
    )

    lr_spec = ModelSpec(
        name="LogisticRegression",
        estimator=LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        needs_scaling=True,
    )

    return (xgb_spec, lgbm_spec, lr_spec)


def evaluate_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train a model with TimeSeriesSplit CV and evaluate on test set."""
    logger.info("Training %s...", spec.name)

    # Scale if needed (logistic regression)
    scaler = None
    X_train_fit = X_train
    X_test_eval = X_test

    if spec.needs_scaling:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_eval = scaler.transform(X_test)

    # Cross-validation on training set
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_results = cross_validate(
        spec.estimator,
        X_train_fit,
        y_train,
        cv=tscv,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=-1,
    )

    cv_accuracy = float(np.mean(cv_results["test_accuracy"]))
    cv_f1 = float(np.mean(cv_results["test_f1"]))
    cv_auc = float(np.mean(cv_results["test_roc_auc"]))

    logger.info(
        "%s CV results - Accuracy: %.4f, F1: %.4f, AUC: %.4f",
        spec.name,
        cv_accuracy,
        cv_f1,
        cv_auc,
    )

    # Train on full training set
    model = spec.estimator
    model.fit(X_train_fit, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_eval)
    y_proba = model.predict_proba(X_test_eval)[:, 1]

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_precision = float(precision_score(y_test, y_pred))
    test_recall = float(recall_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred))
    test_auc = float(roc_auc_score(y_test, y_proba))

    baseline = float(max(y_test.mean(), 1 - y_test.mean()))

    logger.info(
        "%s Test results - Accuracy: %.4f (baseline: %.4f), F1: %.4f, AUC: %.4f",
        spec.name,
        test_accuracy,
        baseline,
        test_f1,
        test_auc,
    )

    # Save model bundle for backtesting
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "metadata": {
            "features": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "needs_scaling": spec.needs_scaling,
        },
    }

    bundle_path = os.path.join(MODELS_DIR, f"model_{spec.name.lower()}.joblib")
    joblib.dump(model_bundle, bundle_path)
    logger.info("Saved %s bundle to %s", spec.name, bundle_path)

    return {
        "model_name": spec.name,
        "model_path": bundle_path,
        "needs_scaling": spec.needs_scaling,
        "cv": {
            "n_splits": N_CV_SPLITS,
            "accuracy_mean": cv_accuracy,
            "accuracy_std": float(np.std(cv_results["test_accuracy"])),
            "f1_mean": cv_f1,
            "f1_std": float(np.std(cv_results["test_f1"])),
            "auc_mean": cv_auc,
            "auc_std": float(np.std(cv_results["test_roc_auc"])),
        },
        "test": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "auc": test_auc,
            "baseline": baseline,
            "improvement_over_baseline": test_accuracy - baseline,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model comparison: XGBoost vs LightGBM vs LogReg")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/training_data_v2.csv",
        help="Input training data CSV",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info("Input: %s", args.input)

    # Load data
    df = pd.read_csv(args.input)
    logger.info("Loaded %d samples with columns: %s", len(df), list(df.columns))

    # Verify feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].values
    y = df["target"].values

    # Time-based train/test split
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # Build and evaluate models
    specs = build_model_specs()
    results = []

    for spec in specs:
        result = evaluate_model(spec, X_train, y_train, X_test, y_test)
        results.append(result)

    # Summary comparison
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    best_by_f1 = max(results, key=lambda r: r["test"]["f1"])
    best_by_auc = max(results, key=lambda r: r["test"]["auc"])

    for r in results:
        logger.info(
            "%-20s | Acc: %.4f | F1: %.4f | AUC: %.4f | Prec: %.4f | Recall: %.4f",
            r["model_name"],
            r["test"]["accuracy"],
            r["test"]["f1"],
            r["test"]["auc"],
            r["test"]["precision"],
            r["test"]["recall"],
        )

    logger.info("Best by F1:  %s (%.4f)", best_by_f1["model_name"], best_by_f1["test"]["f1"])
    logger.info("Best by AUC: %s (%.4f)", best_by_auc["model_name"], best_by_auc["test"]["auc"])

    # Save comparison results
    os.makedirs(MODELS_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.input,
        "samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURE_COLUMNS,
        "models": results,
        "best_by_f1": best_by_f1["model_name"],
        "best_by_auc": best_by_auc["model_name"],
    }

    output_path = os.path.join(MODELS_DIR, "model_comparison.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
