"""VAPM - Verifiable AI Portfolio Manager API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_settings
from db import init_db, close_db
from services import market_data_service, feature_engine
from events import event_publisher

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # ─────────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────────
    print(f"Starting {settings.agent_name}...")

    # Initialize database
    print("Initializing database...")
    await init_db()

    # Start market data service
    print("Starting market data service...")
    await market_data_service.start()

    print(f"{settings.agent_name} is ready!")

    yield

    # ─────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────
    print("Shutting down...")

    await market_data_service.stop()
    await close_db()

    print("Shutdown complete")


app = FastAPI(
    title="VAPM - Verifiable AI Portfolio Manager",
    description="AI trading agent with on-chain verifiable decisions",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# HEALTH & STATUS ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": settings.agent_name,
    }


@app.get("/agent/status")
async def agent_status():
    """Get current agent status."""
    return {
        "agent_name": settings.agent_name,
        "status": "running",
        "latest_price": market_data_service.latest_price,
        "symbol": market_data_service.symbol,
        "position": None,
        "pnl_today": 0.0,
        "total_pnl": 0.0,
        "trades_today": 0,
    }


# ─────────────────────────────────────────────────────────────
# MARKET DATA ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/market/price")
async def get_current_price():
    """Get current price."""
    return {
        "symbol": market_data_service.symbol,
        "price": market_data_service.latest_price,
    }


@app.get("/market/candles")
async def get_candles(limit: int = 100):
    """Get recent candles."""
    candles = await market_data_service.get_recent_candles(limit=limit)
    return {
        "symbol": market_data_service.symbol,
        "interval": "1m",
        "count": len(candles),
        "candles": [
            {
                "open_time": c.open_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ],
    }


@app.get("/market/latest")
async def get_latest_market_data():
    """Get latest market data snapshot."""
    candle = market_data_service.latest_candle
    return {
        "symbol": market_data_service.symbol,
        "price": market_data_service.latest_price,
        "candle": {
            "open": candle.open if candle else None,
            "high": candle.high if candle else None,
            "low": candle.low if candle else None,
            "close": candle.close if candle else None,
            "volume": candle.volume if candle else None,
            "is_closed": candle.is_closed if candle else None,
        } if candle else None,
    }


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINE ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/features/compute")
async def compute_features():
    """Compute and return current technical indicators."""
    features = await feature_engine.compute_features()
    if features is None:
        return {"error": "Not enough data to compute features"}
    return {
        "symbol": market_data_service.symbol,
        "features": features.to_dict(),
        "feature_array": features.to_array().tolist(),
    }


@app.get("/features/latest")
async def get_latest_features():
    """Get most recently computed features."""
    if feature_engine.latest_features is None:
        # Compute if not available
        features = await feature_engine.compute_features()
        if features is None:
            return {"error": "Not enough data to compute features"}
    return {
        "symbol": market_data_service.symbol,
        "features": feature_engine.latest_features.to_dict(),
    }


# ─────────────────────────────────────────────────────────────
# TODO: Add more routes
# ─────────────────────────────────────────────────────────────
# from api.routes import router
# app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
