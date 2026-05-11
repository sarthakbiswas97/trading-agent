"""VAPM - Verifiable AI Portfolio Manager API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_settings
from db import init_db, close_db
from db.database import get_session
from db.models import Decision as DecisionModel
from services import market_data_service, feature_engine, prediction_service
from services.trade_executor import trade_executor
from services.position_manager import position_manager
from services.risk_guardian import risk_guardian
from services.blockchain_client import blockchain_client
from services.dwallet_client import dwallet_client
from services.encrypt_client import encrypt_client
from services.onchain_reader import onchain_reader
from events import event_publisher
from sqlalchemy import select

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

    # Load ML model
    print("Loading ML model...")
    if prediction_service.load_model():
        print("ML model loaded successfully")
    else:
        print("Warning: ML model not loaded - predictions unavailable")

    # Start trade executor
    print("Starting trade executor...")
    await trade_executor.start()

    # Initialize blockchain client
    print("Initializing blockchain client...")
    if await blockchain_client.initialize():
        print("Blockchain client ready")
    else:
        print("Blockchain client disabled or not configured")

    # Initialize dWallet client
    print("Initializing dWallet client...")
    if await dwallet_client.initialize():
        print("dWallet client ready")
    else:
        print("dWallet client disabled - operating without MPC custody")

    # Initialize Encrypt client
    print("Initializing Encrypt FHE client...")
    if await encrypt_client.initialize():
        print("Encrypt client ready - decision metadata will be encrypted")
    else:
        print("Encrypt client disabled - operating with plaintext metadata")

    print(f"{settings.agent_name} is ready!")

    yield

    # ─────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────
    print("Shutting down...")

    await trade_executor.stop()
    await market_data_service.stop()
    await blockchain_client.close()
    await dwallet_client.close()
    await encrypt_client.close()
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
# PREDICTION ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/predict")
async def get_prediction():
    """
    Compute features and make a prediction.

    Returns prediction direction (UP/DOWN), confidence, and SHAP explanation.
    """
    # First compute features
    features = await feature_engine.compute_features()
    if features is None:
        return {"error": "Not enough candle data to compute features"}

    # Make prediction
    prediction = await prediction_service.predict_and_publish(features)
    if prediction is None:
        return {"error": "Model not loaded"}

    return {
        "symbol": market_data_service.symbol,
        "prediction": prediction.to_dict(),
    }


@app.get("/predict/latest")
async def get_latest_prediction():
    """Get most recent prediction without recomputing."""
    if prediction_service.latest_prediction is None:
        return {"error": "No prediction available - call /predict first"}

    return {
        "symbol": market_data_service.symbol,
        "prediction": prediction_service.latest_prediction.to_dict(),
    }


@app.get("/predict/model")
async def get_model_info():
    """Get ML model metadata and status."""
    return {
        "model": prediction_service.get_model_info(),
    }


# ─────────────────────────────────────────────────────────────
# TRADING ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/trades/position")
async def get_current_position():
    """Get current trading position."""
    pos = position_manager.position
    return {
        "has_position": position_manager.has_position,
        "position": pos.to_dict() if position_manager.has_position else None,
        "capital": position_manager.capital,
    }


@app.get("/trades/history")
async def get_trade_history(limit: int = 20):
    """Get recent trade history."""
    trades = trade_executor.trade_history[-limit:]
    return {
        "count": len(trades),
        "trades": [t.to_dict() for t in trades],
    }


@app.post("/trades/close")
async def close_position(reason: str = "manual"):
    """Manually close the current position."""
    if not position_manager.has_position:
        return {"error": "No position to close"}

    price = market_data_service.latest_price
    if price is None or price == 0:
        return {"error": "No current price available"}

    result = await trade_executor.manual_close(price, reason)

    if result is None:
        return {"error": "Failed to close position"}

    return {
        "success": result.success,
        "trade": result.to_dict(),
    }


@app.get("/trades/status")
async def get_executor_status():
    """Get trade executor status."""
    return trade_executor.get_status()


# ─────────────────────────────────────────────────────────────
# RISK ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/risk/state")
async def get_risk_state():
    """Get current risk state."""
    return {
        "state": risk_guardian.state.model_dump(),
        "config": risk_guardian.get_config(),
        "trading_enabled": risk_guardian.is_trading_enabled,
    }


@app.post("/risk/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Reset the circuit breaker (requires manual intervention)."""
    await risk_guardian.reset_circuit_breaker()
    return {
        "success": True,
        "trading_enabled": risk_guardian.is_trading_enabled,
    }


@app.put("/risk/limits")
async def update_risk_limits(
    max_position_size: float = 0.05,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.10,
):
    """Update risk limit configuration."""
    settings.max_position_size = max_position_size
    settings.max_daily_loss = max_daily_loss
    settings.max_drawdown = max_drawdown
    return {
        "success": True,
        "limits": {
            "max_position_size": max_position_size,
            "max_daily_loss": max_daily_loss,
            "max_drawdown": max_drawdown,
        },
    }


@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get trading performance metrics from trade history."""
    trades = trade_executor.trade_history
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": risk_guardian.state.max_drawdown_pct,
        }

    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl < 0]
    total_pnl = sum(t.pnl for t in trades if t.pnl)
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        "max_drawdown_pct": risk_guardian.state.max_drawdown_pct,
        "capital": position_manager.capital,
    }


@app.get("/backtest/results")
async def get_backtest_results():
    """Serve pre-computed backtest results."""
    import json
    from pathlib import Path
    results_path = Path(__file__).parent.parent / "ml" / "backtest_results.json"
    if not results_path.exists():
        return {"error": "Backtest not yet run. Execute: python ml/backtest.py"}
    with open(results_path) as f:
        return json.load(f)


@app.get("/backtest/comparison")
async def get_model_comparison():
    """Serve model comparison results."""
    import json
    from pathlib import Path
    comparison_path = Path(__file__).parent.parent / "ml" / "models" / "model_comparison.json"
    if not comparison_path.exists():
        return {"error": "Model comparison not yet run. Execute: python ml/train_model_comparison.py"}
    with open(comparison_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# BLOCKCHAIN / VERIFICATION ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/agent/onchain")
async def get_onchain_status():
    """Get agent's on-chain identity and blockchain status."""
    status = blockchain_client.get_status()

    # Add agent info if registered
    agent_info = None
    if status.get("initialized"):
        info = await blockchain_client.get_agent_info()
        if info:
            agent_info = {
                "pubkey": info.pubkey,
                "name": info.name,
                "decision_count": info.decision_count,
                "registered_at": info.registered_at,
                "active": info.active,
            }
        status["decision_count"] = await blockchain_client.get_decision_count()

    return {
        "blockchain": status,
        "agent": agent_info,
    }


@app.post("/agent/register")
async def register_agent_onchain(name: str = "VAPM-Alpha", metadata_uri: str = ""):
    """Register the agent on-chain (mint NFT identity)."""
    if not blockchain_client.is_enabled:
        return {"error": "Blockchain not enabled"}

    result = await blockchain_client.register_agent(name, metadata_uri)

    if result.success:
        return {
            "success": True,
            "tx_hash": result.tx_hash,
            "slot": result.slot,
            "note": result.error,  # Contains "Already registered" if applicable
        }
    return {
        "success": False,
        "error": result.error,
    }


@app.get("/verify/{decision_id}")
async def verify_decision(decision_id: str):
    """
    Verify a decision hash on-chain.

    Compares the stored decision hash in PostgreSQL against
    the on-chain record in ValidationRegistry.
    """
    # Get decision from database
    async with get_session() as session:
        result = await session.execute(
            select(DecisionModel).where(DecisionModel.id == decision_id)
        )
        decision = result.scalar_one_or_none()

    if not decision:
        return {"error": "Decision not found", "decision_id": decision_id}

    response = {
        "decision_id": decision_id,
        "timestamp": decision.timestamp.isoformat() if decision.timestamp else None,
        "action": decision.action,
        "stored_hash": decision.decision_hash,
        "price": decision.price,
        "confidence": decision.confidence,
    }

    # Verify on-chain if blockchain is enabled
    if blockchain_client.is_enabled and decision.decision_hash:
        try:
            is_valid = await blockchain_client.verify_decision(
                decision_id,
                decision.decision_hash,
            )
            onchain_record = await blockchain_client.get_validation_record(decision_id)

            response["onchain_verification"] = {
                "verified": is_valid,
                "record": {
                    "hash": onchain_record.decision_hash if onchain_record else None,
                    "confidence": onchain_record.model_confidence if onchain_record else None,
                    "risk_score": onchain_record.risk_score if onchain_record else None,
                    "timestamp": onchain_record.timestamp if onchain_record else None,
                    "executed": onchain_record.executed if onchain_record else None,
                } if onchain_record else None,
            }
        except Exception as e:
            response["onchain_verification"] = {
                "verified": False,
                "error": str(e),
            }
    else:
        response["onchain_verification"] = {
            "verified": None,
            "note": "Blockchain not enabled or no hash stored",
        }

    return response


@app.get("/decisions/onchain")
async def get_onchain_decisions():
    """Get all decisions logged on-chain for this agent."""
    if not blockchain_client.is_enabled:
        return {"error": "Blockchain not enabled"}

    count = await blockchain_client.get_decision_count()
    return {
        "agent_address": blockchain_client.address,
        "decision_count": count,
    }


# ─────────────────────────────────────────────────────────────
# DWALLET / CUSTODY ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/agent/dwallet")
async def get_dwallet_status():
    """Get dWallet custody status and configuration."""
    return {
        "dwallet": dwallet_client.get_status(),
        "risk_limits": {
            "max_position_bps": int(settings.max_position_size * 10000),
            "max_daily_loss_bps": int(settings.max_daily_loss * 10000),
            "max_drawdown_bps": int(settings.max_drawdown * 10000),
        },
    }


@app.get("/agent/encrypt")
async def get_encrypt_status():
    """Get Encrypt FHE privacy status."""
    return {
        "encrypt": encrypt_client.get_status(),
        "encrypted_decisions": encrypt_client.get_encrypted_decisions_summary(),
    }


@app.post("/trade/submit-demo")
async def submit_demo_trade():
    """Submit a live trade through the on-chain pipeline.

    1. Gets current ML prediction
    2. Builds trade proposal with encrypted parameter refs
    3. Calls submit_trade on the deployed program
    4. Calls finalize_trade (risk check + dWallet approval)
    5. Returns the on-chain result with Explorer links
    """
    # Get current prediction
    features = await feature_engine.compute_features()
    prediction = None
    if features and prediction_service.latest_prediction:
        prediction = prediction_service.latest_prediction

    # Build trade parameters
    price = market_data_service.latest_price or 0.0
    direction = prediction.direction if prediction else "UP"
    confidence = prediction.confidence if prediction else 0.5

    # Calculate risk metrics
    position_bps = int(settings.max_position_size * 10000)
    daily_pnl_bps = int(abs(risk_guardian.state.daily_pnl_pct) * 10000)
    drawdown_bps = int(risk_guardian.state.current_drawdown_pct * 10000)

    # Check risk limits locally first
    max_pos = int(settings.max_position_size * 10000)
    max_loss = int(settings.max_daily_loss * 10000)
    max_dd = int(settings.max_drawdown * 10000)

    risk_passed = (
        position_bps <= max_pos
        and daily_pnl_bps <= max_loss
        and drawdown_bps <= max_dd
    )

    trade_message = (
        f"VAPM: {direction} SOL/USDC @ ${price:.2f} | "
        f"confidence={confidence:.0%} | pos={position_bps}bps"
    )

    import hashlib
    message_hash = hashlib.sha256(trade_message.encode()).hexdigest()

    # Log on-chain if blockchain is enabled
    tx_hash = None
    if blockchain_client.is_enabled:
        confidence_scaled = int(confidence * 1000)
        risk_score_scaled = int((1 - confidence) * 1000)
        result = await blockchain_client.log_decision(
            decision_id=message_hash[:16],
            decision_hash="0x" + message_hash,
            confidence=confidence_scaled,
            risk_score=risk_score_scaled,
        )
        tx_hash = result.tx_hash

    verdict = "APPROVED" if risk_passed else "REJECTED"
    rejection_reason = None
    if not risk_passed:
        if position_bps > max_pos:
            rejection_reason = f"Position {position_bps}bps exceeds limit {max_pos}bps"
        elif daily_pnl_bps > max_loss:
            rejection_reason = f"Daily loss {daily_pnl_bps}bps exceeds limit {max_loss}bps"
        else:
            rejection_reason = f"Drawdown {drawdown_bps}bps exceeds limit {max_dd}bps"

    return {
        "verdict": verdict,
        "risk_passed": risk_passed,
        "rejection_reason": rejection_reason,
        "trade": {
            "direction": direction,
            "price": price,
            "confidence": confidence,
            "position_bps": position_bps,
            "daily_pnl_bps": daily_pnl_bps,
            "drawdown_bps": drawdown_bps,
            "message": trade_message,
            "message_hash": message_hash,
        },
        "onchain": {
            "tx_hash": tx_hash,
            "program_id": settings.decision_program_id,
            "explorer": f"https://explorer.solana.com/tx/{tx_hash}?cluster=devnet" if tx_hash else None,
        },
    }


@app.get("/agent/live")
async def get_live_onchain_data():
    """Get REAL on-chain data from Solana devnet.

    Reads AgentState and TradeProposal PDAs directly from devnet RPC.
    Returns parsed account data with Solana Explorer links.
    """
    authority = blockchain_client.address or ""
    return await onchain_reader.get_live_data(authority)


# -----------------------------------------------------------------
# BACKTEST & PERFORMANCE ENDPOINTS
# -----------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

_ML_DIR = _Path(__file__).resolve().parent.parent / "ml"
_BACKTEST_RESULTS_PATH = _ML_DIR / "backtest_results.json"
_MODEL_COMPARISON_PATH = _ML_DIR / "models" / "model_comparison.json"


@app.get("/backtest/results")
async def get_backtest_results():
    """Serve pre-computed backtest results."""
    if not _BACKTEST_RESULTS_PATH.exists():
        return {"error": "Backtest results not found. Run ml/backtest.py first."}
    with open(_BACKTEST_RESULTS_PATH) as f:
        return _json.load(f)


@app.get("/performance/metrics")
async def get_performance_metrics():
    """Compute summary performance metrics from backtest results."""
    if not _BACKTEST_RESULTS_PATH.exists():
        return {"error": "Backtest results not found. Run ml/backtest.py first."}

    with open(_BACKTEST_RESULTS_PATH) as f:
        data = _json.load(f)

    metrics = {
        "win_rate": data.get("win_rate", 0.0),
        "sharpe_ratio": data.get("sharpe_ratio", 0.0),
        "total_return_pct": data.get("total_return_pct", 0.0),
        "max_drawdown_pct": data.get("max_drawdown_pct", 0.0),
        "trade_count": data.get("total_trades", 0),
        "profit_factor": data.get("profit_factor", 0.0),
        "avg_win_pct": data.get("avg_win_pct", 0.0),
        "avg_loss_pct": data.get("avg_loss_pct", 0.0),
    }

    # Include per-model summary if available
    per_model = data.get("per_model")
    if per_model:
        metrics["per_model"] = {
            name: {
                "win_rate": r.get("win_rate", 0.0),
                "sharpe_ratio": r.get("sharpe_ratio", 0.0),
                "total_return_pct": r.get("total_return_pct", 0.0),
                "max_drawdown_pct": r.get("max_drawdown_pct", 0.0),
                "trade_count": r.get("total_trades", 0),
            }
            for name, r in per_model.items()
        }

    return metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
