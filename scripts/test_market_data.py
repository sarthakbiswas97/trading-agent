#!/usr/bin/env python3
"""Test script for market data service - runs standalone without full server."""

import asyncio
import aiohttp
from datetime import datetime


async def test_binance_connection():
    """Test basic Binance API connectivity."""
    print("🔍 Testing Binance API connection...")

    url = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDC"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✅ Binance API OK - ETH/USDC price: ${float(data['price']):,.2f}")
                return True
            else:
                print(f"❌ Binance API error: {resp.status}")
                return False


async def test_websocket_stream():
    """Test Binance WebSocket for 10 seconds."""
    print("\n🔌 Testing Binance WebSocket (10 seconds)...")

    ws_url = "wss://stream.binance.com:9443/ws/ethusdc@trade"
    tick_count = 0

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print("✅ WebSocket connected")

                start_time = asyncio.get_event_loop().time()
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        import json
                        data = json.loads(msg.data)
                        tick_count += 1
                        price = float(data['p'])
                        qty = float(data['q'])

                        if tick_count <= 3:  # Show first 3 ticks
                            print(f"   Tick {tick_count}: ${price:,.2f} x {qty:.4f} ETH")

                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > 10:
                        break

        print(f"✅ Received {tick_count} ticks in 10 seconds")
        print(f"   Average: {tick_count/10:.1f} ticks/second")
        return True

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False


async def test_historical_fetch():
    """Test fetching historical candles."""
    print("\n📊 Testing historical candle fetch...")

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "ETHUSDC",
        "interval": "1m",
        "limit": 10,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✅ Fetched {len(data)} candles")

                # Show last candle
                last = data[-1]
                open_time = datetime.fromtimestamp(last[0] / 1000)
                print(f"   Last candle: {open_time.strftime('%H:%M')}")
                print(f"   O: ${float(last[1]):,.2f} H: ${float(last[2]):,.2f} L: ${float(last[3]):,.2f} C: ${float(last[4]):,.2f}")
                print(f"   Volume: {float(last[5]):,.2f} ETH")
                return True
            else:
                print(f"❌ API error: {resp.status}")
                return False


async def main():
    print("=" * 50)
    print("VAPM Market Data Service Test")
    print("=" * 50)
    print("")

    results = []

    # Test 1: REST API
    results.append(await test_binance_connection())

    # Test 2: Historical data
    results.append(await test_historical_fetch())

    # Test 3: WebSocket
    results.append(await test_websocket_stream())

    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
