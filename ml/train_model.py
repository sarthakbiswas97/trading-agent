"""
XGBoost Model Training for Price Direction Prediction.

This script:
1. Loads prepared training data
2. Splits data by time (train/val/test)
3. Creates sklearn Pipeline (preprocessing + model)
4. Trains XGBoost classifier
5. Evaluates performance
6. Saves versioned model and artifacts

Best Practices:
- Pipeline includes preprocessing (StandardScaler)
- Models are versioned (never overwritten)
- All metadata saved for reproducibility
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# XGBoost (sklearn API)
from xgboost import XGBClassifier

# SHAP for explainability
import shap


# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE = 'dataset/training_data.csv'

# Model versioning - creates new version each run
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = 'models'
MODEL_OUTPUT = f'{MODELS_DIR}/model_bundle_v{MODEL_VERSION}.joblib'
METRICS_OUTPUT = f'{MODELS_DIR}/metrics_v{MODEL_VERSION}.json'

# Also save as "latest" for easy access
LATEST_MODEL = f'{MODELS_DIR}/model_bundle_latest.joblib'
LATEST_METRICS = f'{MODELS_DIR}/metrics_latest.json'

# Features to use (must match data_preparation.py)
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
TRAIN_RATIO = 0.8  # 80% for training (includes validation)
VAL_RATIO = 0.1    # 10% of train for early stopping validation

# XGBoost parameters
EARLY_STOPPING_ROUNDS = 10  # Stop if no improvement for 10 rounds

XGB_PARAMS = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 4,             # Tree depth (prevent overfitting)
    'learning_rate': 0.1,       # Step size
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',   # Evaluation metric
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS,  # In constructor for sklearn compat
    'random_state': 42,         # Reproducibility
    'n_jobs': -1,               # Use all CPU cores
}


# ============================================
# STEP 1: Load Data
# ============================================

print("=" * 60)
print("Step 1: Loading data...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} samples")
print(f"Columns: {list(df.columns)}")

# Separate features and target
X = df[FEATURE_COLUMNS].values
y = df['target'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution: {np.bincount(y)} (0=down, 1=up)")


# ============================================
# STEP 2: Time-Based Train/Test Split
# ============================================

print("\n" + "=" * 60)
print("Step 2: Splitting data (time-based)...")
print("=" * 60)

# Split point for train/test
split_idx = int(len(df) * TRAIN_RATIO)

# Time-based split (no shuffling!)
X_train_full = X[:split_idx]
y_train_full = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"Train+Val samples: {len(X_train_full)} (first {TRAIN_RATIO*100:.0f}%)")
print(f"Test samples: {len(X_test)} (last {(1-TRAIN_RATIO)*100:.0f}%)")

# Further split train into train/validation for early stopping
# Using shuffle=False to maintain time order
val_size = int(len(X_train_full) * VAL_RATIO / TRAIN_RATIO)
X_train = X_train_full[:-val_size]
y_train = y_train_full[:-val_size]
X_val = X_train_full[-val_size:]
y_val = y_train_full[-val_size:]

print(f"\nFinal split:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")


# ============================================
# STEP 3: Create Pipeline (Preprocessing + Model)
# ============================================

print("\n" + "=" * 60)
print("Step 3: Creating pipeline...")
print("=" * 60)

# Pipeline: StandardScaler → XGBClassifier
# This ensures preprocessing is saved with the model!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(**XGB_PARAMS)),
])

print("Pipeline structure:")
print(pipeline)


# ============================================
# STEP 4: Train Model
# ============================================

print("\n" + "=" * 60)
print("Step 4: Training XGBoost model...")
print("=" * 60)

print(f"XGBoost parameters: {XGB_PARAMS}")
print(f"Model version: {MODEL_VERSION}")

# Fit scaler on train data first
scaler = pipeline.named_steps['scaler']
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost with early stopping
model = pipeline.named_steps['model']
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True,
)

print(f"\nTraining complete!")
print(f"Best iteration: {model.best_iteration}")


# ============================================
# STEP 5: Evaluate Model
# ============================================

print("\n" + "=" * 60)
print("Step 5: Evaluating model...")
print("=" * 60)

# Predictions on test set (already scaled)
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (up)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n=== Test Set Results ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Baseline comparison
baseline = max(np.mean(y_test), 1 - np.mean(y_test))
print(f"\nBaseline (always predict majority): {baseline:.4f}")
print(f"Model improvement over baseline: {(accuracy - baseline)*100:+.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 DOWN    UP")
print(f"Actual DOWN    {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"Actual UP      {cm[1,0]:5d}  {cm[1,1]:5d}")

# Detailed classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))


# ============================================
# STEP 6: Feature Importance
# ============================================

print("\n" + "=" * 60)
print("Step 6: Feature importance...")
print("=" * 60)

# Get feature importance from model
importance = model.feature_importances_

# Create sorted list
feature_importance = sorted(
    zip(FEATURE_COLUMNS, importance),
    key=lambda x: x[1],
    reverse=True
)

print("\nFeature Importance (higher = more predictive):")
for feature, imp in feature_importance:
    bar = "█" * int(imp * 50)
    print(f"  {feature:20s}: {imp:.4f} {bar}")


# ============================================
# STEP 7: SHAP Explainability (Sample)
# ============================================

print("\n" + "=" * 60)
print("Step 7: SHAP explainability (sample predictions)...")
print("=" * 60)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values for a few test samples
sample_size = 5
X_sample = X_test_scaled[:sample_size]
shap_values = explainer.shap_values(X_sample)

print(f"\nSample SHAP explanations (first {sample_size} test predictions):")
for i in range(sample_size):
    pred_proba = y_pred_proba[i]
    actual = y_test[i]
    pred = y_pred[i]

    print(f"\n--- Sample {i+1} ---")
    print(f"Prediction: {'UP' if pred == 1 else 'DOWN'} (confidence: {pred_proba:.2%})")
    print(f"Actual: {'UP' if actual == 1 else 'DOWN'}")
    print(f"Correct: {'✓' if pred == actual else '✗'}")
    print(f"Top contributing features:")

    # Sort SHAP values for this sample
    sample_shap = list(zip(FEATURE_COLUMNS, shap_values[i]))
    sample_shap_sorted = sorted(sample_shap, key=lambda x: abs(x[1]), reverse=True)

    for feature, shap_val in sample_shap_sorted[:3]:
        direction = "↑ pushes UP" if shap_val > 0 else "↓ pushes DOWN"
        print(f"    {feature}: {shap_val:+.4f} {direction}")


# ============================================
# STEP 8: Save Model Bundle (Model + Metadata)
# ============================================

print("\n" + "=" * 60)
print("Step 8: Saving model bundle (versioned)...")
print("=" * 60)

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

# Create metadata - everything needed to use the model
metadata = {
    'version': MODEL_VERSION,
    'timestamp': datetime.now().isoformat(),
    'features': FEATURE_COLUMNS,
    'feature_count': len(FEATURE_COLUMNS),
    'dataset': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
    },
    'parameters': XGB_PARAMS,
    'training': {
        'best_iteration': model.best_iteration,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    },
    'results': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'baseline': float(baseline),
        'improvement_over_baseline': float(accuracy - baseline),
    },
    'confusion_matrix': {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1]),
    },
    'feature_importance': {feat: float(imp) for feat, imp in feature_importance},
}

# Bundle everything together - pipeline + metadata in ONE file
model_bundle = {
    'pipeline': pipeline,      # Includes scaler + model
    'metadata': metadata,      # All metadata
}

# Save versioned bundle
joblib.dump(model_bundle, MODEL_OUTPUT)
print(f"Model bundle saved to: {MODEL_OUTPUT}")

# Also save as latest for easy access
joblib.dump(model_bundle, LATEST_MODEL)
print(f"Latest bundle: {LATEST_MODEL}")

# Also save metrics as JSON for easy reading without loading model
with open(METRICS_OUTPUT, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metrics JSON: {METRICS_OUTPUT}")

with open(LATEST_METRICS, 'w') as f:
    json.dump(metadata, f, indent=2)


# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 60)
print(f"""
Model Version: {MODEL_VERSION}

Dataset:
  Total: {len(df)} samples
  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}

Model Performance:
  Accuracy:    {accuracy*100:.2f}%
  Baseline:    {baseline*100:.2f}%
  Improvement: {(accuracy-baseline)*100:+.2f}%

Top 3 Predictive Features:
  1. {feature_importance[0][0]}: {feature_importance[0][1]:.4f}
  2. {feature_importance[1][0]}: {feature_importance[1][1]:.4f}
  3. {feature_importance[2][0]}: {feature_importance[2][1]:.4f}

Saved Files:
  - {MODEL_OUTPUT} (versioned bundle)
  - {LATEST_MODEL} (latest bundle)
  - {METRICS_OUTPUT} (metrics JSON)

How to Load & Use:
  # Load bundle
  bundle = joblib.load('{LATEST_MODEL}')
  pipeline = bundle['pipeline']
  metadata = bundle['metadata']

  # Check model info
  print(f"Model version: {{metadata['version']}}")
  print(f"Accuracy: {{metadata['results']['accuracy']}}")
  print(f"Features: {{metadata['features']}}")

  # Make prediction
  prediction = pipeline.predict(features)
  probability = pipeline.predict_proba(features)[:, 1]
""")
