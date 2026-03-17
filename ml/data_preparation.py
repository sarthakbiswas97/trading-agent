"""
Data preparation for XGBoost training.

This script:
1. Loads historical candles from CSV
2. Computes features for each candle using sliding window
3. Creates target labels (price up/down in N minutes)
4. Saves training-ready dataset
"""

import pandas as pd
import numpy as np
import sys
import os

# Add backend to path so we can import core module
sys.path.append('../backend')

from core.indicators import compute_all_features, normalize_features, MIN_CANDLES, FEATURE_NAMES


# ============================================
# CONFIGURATION
# ============================================

PREDICTION_HORIZON = 5  # Predict price movement 5 candles ahead
INPUT_FILE = 'candles.csv'
OUTPUT_FILE = 'dataset/training_data.csv'


# ============================================
# STEP 1: Load candles
# ============================================

print("=" * 50)
print("Step 1: Loading candles...")
print("=" * 50)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} candles")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
print(df.head())


# ============================================
# STEP 2: Compute features with sliding window
# ============================================

print("\n" + "=" * 50)
print(f"Step 2: Computing features (window size: {MIN_CANDLES} candles)...")
print("=" * 50)

# We need:
# - At least MIN_CANDLES (35) candles for the first feature computation
# - PREDICTION_HORIZON (5) candles after each point for the target label
#
# So valid range is: index 34 to index (len-5)
#
# Example with 100 candles:
#   - First valid: index 34 (uses candles 0-34)
#   - Last valid: index 94 (needs candle 99 for target)

start_idx = MIN_CANDLES - 1  # First index where we have enough history
end_idx = len(df) - PREDICTION_HORIZON  # Last index where we have future data for target

print(f"Valid index range: {start_idx} to {end_idx}")
print(f"Total samples to generate: {end_idx - start_idx}")

# Store features and metadata
features_list = []
timestamps = []
prices = []

# Progress tracking
total = end_idx - start_idx
progress_step = total // 10  # Print progress every 10%

for i in range(start_idx, end_idx):
    # Progress update
    if (i - start_idx) % progress_step == 0:
        pct = (i - start_idx) / total * 100
        print(f"  Progress: {pct:.0f}% ({i - start_idx}/{total})")

    # Get window of candles ending at index i
    # Window includes candles from (i - MIN_CANDLES + 1) to i (inclusive)
    window_start = i - MIN_CANDLES + 1
    window_end = i + 1  # +1 because iloc is exclusive on end

    window = df.iloc[window_start:window_end]

    # Extract closes and volumes as numpy arrays
    closes = window['close'].values
    volumes = window['volume'].values

    # Compute features
    features = compute_all_features(closes, volumes)

    # Store results
    features_list.append(features)
    timestamps.append(df.iloc[i]['open_time'])
    prices.append(df.iloc[i]['close'])

print(f"  Progress: 100% ({total}/{total})")
print(f"Computed features for {len(features_list)} samples")


# ============================================
# STEP 3: Create target labels
# ============================================

print("\n" + "=" * 50)
print("Step 3: Creating target labels...")
print("=" * 50)

# Target = 1 if price goes UP after PREDICTION_HORIZON candles
# Target = 0 if price goes DOWN

targets = []

for i in range(start_idx, end_idx):
    current_price = df.iloc[i]['close']
    future_price = df.iloc[i + PREDICTION_HORIZON]['close']

    # 1 = price went up, 0 = price went down or stayed same
    target = 1 if future_price > current_price else 0
    targets.append(target)

# Count distribution
up_count = sum(targets)
down_count = len(targets) - up_count

print(f"Total samples: {len(targets)}")
print(f"Price UP (target=1): {up_count} ({up_count/len(targets)*100:.1f}%)")
print(f"Price DOWN (target=0): {down_count} ({down_count/len(targets)*100:.1f}%)")


# ============================================
# STEP 4: Save dataset
# ============================================

print("\n" + "=" * 50)
print("Step 4: Saving dataset...")
print("=" * 50)

# Create output directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Build DataFrame from features
# Each row = one sample with all features + target
data_rows = []

for i, (features, timestamp, price, target) in enumerate(zip(features_list, timestamps, prices, targets)):
    row = {
        'timestamp': timestamp,
        'price': price,
        'target': target,
        # Add all features
        'rsi': features['rsi'],
        'macd': features['macd'],
        'macd_signal': features['macd_signal'],
        'macd_histogram': features['macd_histogram'],
        'ema_ratio': features['ema_ratio'],
        'volatility': features['volatility'],
        'volume_spike': features['volume_spike'],
        'momentum': features['momentum'],
        'bollinger_position': features['bollinger_position'],
    }
    data_rows.append(row)

# Create DataFrame
dataset_df = pd.DataFrame(data_rows)

# Save to CSV
dataset_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(dataset_df)} samples to {OUTPUT_FILE}")
print(f"\nDataset shape: {dataset_df.shape}")
print(f"Columns: {list(dataset_df.columns)}")
print(f"\nFirst 5 rows:")
print(dataset_df.head())
print(f"\nFeature statistics:")
print(dataset_df[FEATURE_NAMES].describe())



