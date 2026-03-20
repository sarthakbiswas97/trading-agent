"""
Data preparation v2 - with regime features.

This script:
1. Loads historical candles from CSV
2. Computes features including regime indicators (ADX, ATR, etc.)
3. Creates target labels (price up/down in N minutes)
4. Saves training-ready dataset

Usage:
    python data_preparation_v2.py --input candles_3months.csv --output dataset/training_data_v2.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os

# Add backend to path so we can import core module
sys.path.append('../backend')

from core.indicators import (
    compute_all_features,
    FEATURE_NAMES,
    REGIME_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    MIN_CANDLES,
)


# ============================================
# CONFIGURATION
# ============================================

PREDICTION_HORIZON = 15  # Predict price movement 15 candles ahead (from experiment)


def prepare_data(input_file: str, output_file: str):
    """Prepare training data with regime features."""

    print("=" * 60)
    print("DATA PREPARATION v2 - With Regime Features")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Prediction horizon: {PREDICTION_HORIZON} candles")
    print(f"Min candles needed: {MIN_CANDLES}")
    print("=" * 60)

    # Load candles
    print("\nStep 1: Loading candles...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} candles")
    print(f"Columns: {list(df.columns)}")

    # Verify required columns
    required_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Compute features with sliding window
    print(f"\nStep 2: Computing features (window size: {MIN_CANDLES} candles)...")

    start_idx = MIN_CANDLES - 1
    end_idx = len(df) - PREDICTION_HORIZON

    print(f"Valid index range: {start_idx} to {end_idx}")
    print(f"Total samples to generate: {end_idx - start_idx:,}")

    features_list = []
    timestamps = []
    prices = []
    targets = []

    total = end_idx - start_idx
    progress_step = max(1, total // 10)

    for i in range(start_idx, end_idx):
        # Progress update
        if (i - start_idx) % progress_step == 0:
            pct = (i - start_idx) / total * 100
            print(f"  Progress: {pct:.0f}% ({i - start_idx:,}/{total:,})")

        # Get window of candles
        window_start = i - MIN_CANDLES + 1
        window_end = i + 1

        window = df.iloc[window_start:window_end]

        # Extract arrays
        closes = window['close'].values
        volumes = window['volume'].values
        highs = window['high'].values
        lows = window['low'].values

        # Compute features (including regime features)
        features = compute_all_features(
            closes, volumes, highs, lows,
            include_regime=True
        )

        features_list.append(features)
        timestamps.append(df.iloc[i]['open_time'])
        prices.append(df.iloc[i]['close'])

        # Target: price UP after PREDICTION_HORIZON candles
        current_price = df.iloc[i]['close']
        future_price = df.iloc[i + PREDICTION_HORIZON]['close']
        targets.append(1 if future_price > current_price else 0)

    print(f"  Progress: 100% ({total:,}/{total:,})")
    print(f"Computed features for {len(features_list):,} samples")

    # Create DataFrame
    print("\nStep 3: Creating dataset...")

    # Count distribution
    up_count = sum(targets)
    down_count = len(targets) - up_count
    print(f"Target distribution:")
    print(f"  Price UP (target=1): {up_count:,} ({up_count/len(targets)*100:.1f}%)")
    print(f"  Price DOWN (target=0): {down_count:,} ({down_count/len(targets)*100:.1f}%)")

    # Build rows
    data_rows = []
    for features, timestamp, price, target in zip(features_list, timestamps, prices, targets):
        row = {
            'timestamp': timestamp,
            'price': price,
            'target': target,
        }
        # Add all features
        for name in FEATURE_NAMES:
            row[name] = features[name]
        for name in REGIME_FEATURE_NAMES:
            row[name] = features[name]
        data_rows.append(row)

    dataset_df = pd.DataFrame(data_rows)

    # Save
    print("\nStep 4: Saving dataset...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    dataset_df.to_csv(output_file, index=False)

    print(f"Saved {len(dataset_df):,} samples to {output_file}")
    print(f"\nDataset shape: {dataset_df.shape}")
    print(f"Columns ({len(dataset_df.columns)}): {list(dataset_df.columns)}")

    # Feature statistics
    print(f"\nBase Feature Statistics:")
    print(dataset_df[FEATURE_NAMES].describe().round(4))

    print(f"\nRegime Feature Statistics:")
    print(dataset_df[REGIME_FEATURE_NAMES].describe().round(4))

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    return dataset_df


def main():
    parser = argparse.ArgumentParser(description="Prepare training data with regime features")
    parser.add_argument("--input", type=str, default="candles_3months.csv", help="Input candles CSV")
    parser.add_argument("--output", type=str, default="dataset/training_data_v2.csv", help="Output file")
    args = parser.parse_args()

    prepare_data(args.input, args.output)


if __name__ == "__main__":
    main()
