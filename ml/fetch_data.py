"""
Fetch historical candle data from Binance.

Usage:
    python fetch_data.py --months 3
    python fetch_data.py --months 6 --symbol BTCUSDT
"""

import argparse
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path


# ============================================
# CONFIGURATION
# ============================================

BINANCE_API = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOL = "ETHUSDT"
DEFAULT_INTERVAL = "1m"
CANDLES_PER_REQUEST = 1000  # Binance max
RATE_LIMIT_DELAY = 0.5  # Seconds between requests


def fetch_candles(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> list[dict]:
    """
    Fetch candles from Binance API.

    Args:
        symbol: Trading pair (e.g., 'ETHUSDT')
        interval: Candle interval (e.g., '1m', '5m', '1h')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        List of candle dictionaries
    """
    all_candles = []
    current_start = start_time

    # Calculate total requests needed
    total_minutes = (end_time - start_time) // (60 * 1000)
    total_requests = (total_minutes // CANDLES_PER_REQUEST) + 1

    print(f"Fetching {total_minutes:,} candles in ~{total_requests} requests...")

    request_count = 0

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": CANDLES_PER_REQUEST,
        }

        try:
            response = requests.get(BINANCE_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"  Error fetching data: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if not data:
            break

        for candle in data:
            all_candles.append({
                "open_time": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "close_time": candle[6],
                "quote_volume": float(candle[7]),
                "trades": candle[8],
            })

        # Move to next batch
        current_start = data[-1][0] + 1  # +1 to avoid overlap
        request_count += 1

        # Progress update
        if request_count % 10 == 0:
            pct = min(100, (current_start - start_time) / (end_time - start_time) * 100)
            print(f"  Progress: {pct:.1f}% ({len(all_candles):,} candles)")

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    return all_candles


def validate_candles(df: pd.DataFrame) -> dict:
    """
    Validate candle data quality.

    Returns dict with quality metrics.
    """
    issues = []

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Missing values: {null_counts.to_dict()}")

    # Check for impossible values
    invalid_hloc = df[df['high'] < df['low']]
    if len(invalid_hloc) > 0:
        issues.append(f"High < Low: {len(invalid_hloc)} rows")

    invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
    if len(invalid_close) > 0:
        issues.append(f"Close outside H/L range: {len(invalid_close)} rows")

    # Check for gaps (missing candles)
    df_sorted = df.sort_values('open_time')
    time_diffs = df_sorted['open_time'].diff().dropna()
    expected_diff = 60000  # 1 minute in milliseconds
    gaps = time_diffs[time_diffs > expected_diff * 1.5]
    if len(gaps) > 0:
        issues.append(f"Time gaps detected: {len(gaps)} gaps")

    # Check for duplicates
    duplicates = df.duplicated(subset=['open_time'])
    if duplicates.sum() > 0:
        issues.append(f"Duplicate timestamps: {duplicates.sum()}")

    return {
        "total_rows": len(df),
        "date_range": f"{df['open_time'].min()} to {df['open_time'].max()}",
        "issues": issues,
        "is_valid": len(issues) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance candle data")
    parser.add_argument("--months", type=int, default=3, help="Number of months to fetch")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Trading pair")
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL, help="Candle interval")
    parser.add_argument("--output", type=str, default="candles_extended.csv", help="Output file")
    args = parser.parse_args()

    print("=" * 60)
    print("Binance Historical Data Fetcher")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Months: {args.months}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.months * 30)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    print(f"\nDate range:")
    print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch candles
    print("\n" + "-" * 60)
    candles = fetch_candles(args.symbol, args.interval, start_ms, end_ms)
    print(f"\nFetched {len(candles):,} candles")

    if not candles:
        print("ERROR: No candles fetched!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Validate
    print("\n" + "-" * 60)
    print("Validating data quality...")
    validation = validate_candles(df)

    print(f"  Total rows: {validation['total_rows']:,}")
    print(f"  Date range: {validation['date_range']}")

    if validation['is_valid']:
        print("  Status: VALID")
    else:
        print("  Status: ISSUES FOUND")
        for issue in validation['issues']:
            print(f"    - {issue}")

    # Remove duplicates if any
    df = df.drop_duplicates(subset=['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)

    # Save to CSV (compatible format with existing candles.csv)
    output_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df[output_cols].to_csv(args.output, index=False)

    print("\n" + "-" * 60)
    print(f"Saved {len(df):,} candles to {args.output}")

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Average close: ${df['close'].mean():.2f}")
    print(f"Total volume: {df['volume'].sum():,.2f}")
    print(f"Average volume/candle: {df['volume'].mean():,.2f}")

    # Calculate some regime info
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100
    up_candles = (returns > 0).sum()
    down_candles = (returns < 0).sum()

    print(f"\nMarket regime metrics:")
    print(f"  Volatility (1-min returns std): {volatility:.4f}%")
    print(f"  Up candles: {up_candles:,} ({up_candles/len(returns)*100:.1f}%)")
    print(f"  Down candles: {down_candles:,} ({down_candles/len(returns)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Done! You can now run data_preparation.py with this data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
