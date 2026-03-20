#!/usr/bin/env python3
"""
Test trade execution flow by injecting simulated predictions.

This script:
1. Publishes a high-confidence UP prediction to trigger a BUY
2. Waits for position to open
3. Publishes a DOWN prediction to trigger exit
4. Verifies the full cycle completed
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis.asyncio as redis
from config import get_settings

settings = get_settings()


async def publish_prediction(r: redis.Redis, direction: str, confidence: float, price: float):
    """Publish a simulated prediction event."""
    import time

    prediction = {
        "timestamp": int(time.time() * 1000),
        "price": price,
        "direction": direction,
        "confidence": confidence,
        "shap_explanation": {
            "test_feature": {
                "value": 0.1,
                "direction": f"pushes {direction}"
            }
        }
    }

    await r.publish("event:prediction_ready", json.dumps(prediction))
    print(f"Published: {direction} @ {confidence:.0%} confidence, ${price:.2f}")
    return prediction


async def get_position(r: redis.Redis) -> dict:
    """Get current position from Redis."""
    data = await r.get("position:current")
    if data:
        return json.loads(data)
    return {}


async def get_risk_state(r: redis.Redis) -> dict:
    """Get risk state from Redis."""
    data = await r.get("risk:state")
    if data:
        return json.loads(data)
    return {}


async def main():
    print("="*60)
    print("Trade Execution Flow Test")
    print("="*60)

    # Connect to Redis
    r = redis.from_url(settings.redis_url, decode_responses=True)
    await r.ping()
    print(f"Connected to Redis at {settings.redis_url}")

    # Check initial state
    print("\n[1] Initial State")
    pos = await get_position(r)
    print(f"    Position: {'OPEN' if pos.get('side') else 'NONE'}")

    # If there's an existing position, skip the entry test
    if pos.get('side'):
        print("    Existing position found, testing EXIT flow only")

        # Publish DOWN signal to exit
        print("\n[2] Publishing EXIT signal (DOWN @ 70%)")
        current_price = pos.get('current_price', 2200.0)
        await publish_prediction(r, "DOWN", 0.70, current_price * 0.99)  # Slight loss

        await asyncio.sleep(2)

        pos = await get_position(r)
        print(f"\n[3] After EXIT signal")
        print(f"    Position: {'OPEN' if pos.get('side') else 'CLOSED'}")

    else:
        # Test ENTRY flow
        print("\n[2] Publishing ENTRY signal (UP @ 75%)")
        entry_price = 2200.0
        await publish_prediction(r, "UP", 0.75, entry_price)

        # Wait for trade executor to process
        await asyncio.sleep(2)

        # Check position
        pos = await get_position(r)
        print(f"\n[3] After ENTRY signal")
        if pos.get('side'):
            print(f"    Position: LONG")
            print(f"    Size: {pos.get('size', 0):.6f} ETH")
            print(f"    Entry: ${pos.get('entry_price', 0):.2f}")

            # Test EXIT flow - simulate price increase for take profit
            print("\n[4] Publishing EXIT signal (take profit scenario)")
            exit_price = entry_price * 1.05  # +5% for take profit
            await publish_prediction(r, "UP", 0.50, exit_price)  # Low confidence, but price triggers TP

            await asyncio.sleep(2)

            pos = await get_position(r)
            print(f"\n[5] After price update (waiting for take profit)")
            print(f"    Position: {'OPEN' if pos.get('side') else 'CLOSED'}")

            if pos.get('side'):
                # Force exit with reversal signal
                print("\n[6] Publishing REVERSAL signal (DOWN @ 60%)")
                await publish_prediction(r, "DOWN", 0.60, exit_price)

                await asyncio.sleep(2)

                pos = await get_position(r)
                print(f"\n[7] After REVERSAL signal")
                print(f"    Position: {'OPEN' if pos.get('side') else 'CLOSED'}")
        else:
            print("    Position: NOT OPENED - check trade executor logs")

    # Final state
    print("\n" + "="*60)
    print("Final State")
    print("="*60)

    pos = await get_position(r)
    risk = await get_risk_state(r)

    print(f"Position: {'OPEN' if pos.get('side') else 'NONE'}")
    print(f"Trades today: {risk.get('trades_today', 0)}")
    print(f"Daily PnL: {risk.get('daily_pnl_pct', 0):.2%}")
    print(f"Trading enabled: {risk.get('trading_enabled', True)}")

    await r.close()
    print("\nTest complete. Check backend logs for detailed execution trace.")


if __name__ == "__main__":
    asyncio.run(main())
