"""
Integration tests for Trade Executor Service.

Tests the full flow: prediction → decision → risk check → execution
"""

import pytest
import asyncio
from datetime import datetime, timezone

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.position_manager import PositionManager, PositionState
from services.risk_guardian import RiskGuardian, RiskConfig
from services.trade_executor import TradeExecutorService, ENTRY_CONFIDENCE_THRESHOLD, BASE_POSITION_SIZE_PCT
from models.decision import TradeAction


class MockEventPublisher:
    """Mock Redis publisher for testing."""

    def __init__(self):
        self.published = []
        self.stored = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def publish(self, channel: str, data: dict):
        self.published.append((channel, data))

    async def add_to_stream(self, stream: str, data: dict, maxlen: int = 1000):
        if stream not in self.stored:
            self.stored[stream] = []
        self.stored[stream].append(data)

    async def set_json(self, key: str, data: dict, expire_seconds: int = None):
        self.stored[key] = data

    async def get_json(self, key: str):
        return self.stored.get(key)


class TestPositionManager:
    """Tests for PositionManager."""

    def test_empty_position(self):
        """Test initial empty position state."""
        pm = PositionManager(initial_capital=10000.0)
        assert not pm.has_position
        assert pm.position.size == 0.0
        assert pm.position.side == ""

    def test_calculate_position_size(self):
        """Test position size calculation."""
        pm = PositionManager(initial_capital=10000.0)

        # 3% of $10,000 = $300 worth of ETH at $2000 = 0.15 ETH
        size = pm.calculate_position_size(price=2000.0, position_pct=0.03)
        assert size == 0.15

        # 5% of $10,000 = $500 worth at $2500 = 0.2 ETH
        size = pm.calculate_position_size(price=2500.0, position_pct=0.05)
        assert size == 0.2

    @pytest.mark.asyncio
    async def test_open_close_position(self):
        """Test opening and closing a position."""
        from unittest.mock import patch

        mock_publisher = MockEventPublisher()

        # Patch where it's used (services.position_manager.event_publisher)
        with patch('services.position_manager.event_publisher', mock_publisher):
            pm = PositionManager(initial_capital=10000.0)

            # Open position
            pos = await pm.open_position(
                side="LONG",
                size=0.15,
                entry_price=2000.0,
                decision_id="test-decision-1",
            )

            assert pm.has_position
            assert pos.side == "LONG"
            assert pos.size == 0.15
            assert pos.entry_price == 2000.0

            # Update price and check PnL
            await pm.update_price(2100.0)
            assert pm.position.current_price == 2100.0
            assert pm.position.unrealized_pnl == pytest.approx(15.0)  # 0.15 * 100
            assert pm.position.unrealized_pnl_pct == pytest.approx(0.05)  # 5%

            # Close position
            closed, realized_pnl = await pm.close_position(
                exit_price=2100.0,
                reason="test",
                decision_id="test-decision-2",
            )

            assert not pm.has_position
            assert realized_pnl == pytest.approx(15.0)


class TestRiskGuardian:
    """Tests for RiskGuardian."""

    def test_default_config(self):
        """Test default risk configuration."""
        rg = RiskGuardian()
        assert rg.config.max_position_size_pct == 0.05
        assert rg.config.max_daily_loss_pct == 0.03
        assert rg.config.stop_loss_pct == 0.02
        assert rg.config.take_profit_pct == 0.04

    @pytest.mark.asyncio
    async def test_check_trade_passing(self):
        """Test risk check that should pass."""
        rg = RiskGuardian(config=RiskConfig(
            min_trade_interval_seconds=0,  # No cooldown for test
        ))

        result = await rg.check_trade(
            action="BUY",
            position_size_pct=0.03,
            current_exposure_pct=0.0,
        )

        assert result.can_trade
        assert len(result.violations) == 0
        assert result.checks["position_size"]
        assert result.checks["total_exposure"]

    @pytest.mark.asyncio
    async def test_check_trade_position_too_large(self):
        """Test risk check failing for oversized position."""
        rg = RiskGuardian(config=RiskConfig(
            max_position_size_pct=0.05,
            min_trade_interval_seconds=0,
        ))

        result = await rg.check_trade(
            action="BUY",
            position_size_pct=0.10,  # 10% - too large
            current_exposure_pct=0.0,
        )

        assert not result.can_trade
        assert not result.checks["position_size"]
        assert "Position size" in result.violations[0]

    @pytest.mark.asyncio
    async def test_check_trade_exposure_exceeded(self):
        """Test risk check failing for excessive exposure."""
        rg = RiskGuardian(config=RiskConfig(
            max_total_exposure_pct=0.10,
            min_trade_interval_seconds=0,
        ))

        result = await rg.check_trade(
            action="BUY",
            position_size_pct=0.05,
            current_exposure_pct=0.08,  # Total would be 13%
        )

        assert not result.can_trade
        assert not result.checks["total_exposure"]

    def test_stop_loss_check(self):
        """Test stop loss trigger."""
        rg = RiskGuardian(config=RiskConfig(stop_loss_pct=0.02))

        assert not rg.check_stop_loss(-0.01)  # -1% - not triggered
        assert rg.check_stop_loss(-0.02)      # -2% - triggered
        assert rg.check_stop_loss(-0.05)      # -5% - triggered

    def test_take_profit_check(self):
        """Test take profit trigger."""
        rg = RiskGuardian(config=RiskConfig(take_profit_pct=0.04))

        assert not rg.check_take_profit(0.03)  # +3% - not triggered
        assert rg.check_take_profit(0.04)      # +4% - triggered
        assert rg.check_take_profit(0.10)      # +10% - triggered

    def test_throttle_factor_brackets(self):
        """Test drawdown-based throttling brackets."""
        rg = RiskGuardian()

        # No drawdown = full size
        rg._state.current_drawdown_pct = 0.0
        assert rg.get_throttle_factor() == 1.0

        # 1% drawdown = full size
        rg._state.current_drawdown_pct = 0.01
        assert rg.get_throttle_factor() == 1.0

        # 3% drawdown = 75% size
        rg._state.current_drawdown_pct = 0.03
        assert rg.get_throttle_factor() == 0.75

        # 5% drawdown = 50% size
        rg._state.current_drawdown_pct = 0.05
        assert rg.get_throttle_factor() == 0.50

        # 7% drawdown = 25% size
        rg._state.current_drawdown_pct = 0.07
        assert rg.get_throttle_factor() == 0.25

        # 8%+ drawdown = no trading
        rg._state.current_drawdown_pct = 0.08
        assert rg.get_throttle_factor() == 0.0

    def test_volatility_scaled_position_size(self):
        """Test volatility-based position sizing."""
        rg = RiskGuardian(config=RiskConfig(
            target_volatility=0.02,  # 2% target
            max_position_size_pct=0.05,
        ))
        rg._state.current_drawdown_pct = 0.0  # No throttling

        # Normal volatility = base size
        size = rg.calculate_position_size(base_size_pct=0.03, current_volatility=0.02)
        assert size == pytest.approx(0.03)

        # High volatility (4%) = half size
        size = rg.calculate_position_size(base_size_pct=0.03, current_volatility=0.04)
        assert size == pytest.approx(0.015)

        # Low volatility (1%) = double size (capped at max)
        size = rg.calculate_position_size(base_size_pct=0.03, current_volatility=0.01)
        assert size == pytest.approx(0.05)  # Capped at max 5%

    def test_atr_based_stop_loss(self):
        """Test ATR-based dynamic stop-loss calculation."""
        rg = RiskGuardian(config=RiskConfig(
            atr_stop_multiplier=2.0,
            max_stop_loss_pct=0.03,
        ))

        # Entry at $2000, ATR = $20 → stop distance = $40 (2x ATR)
        # 3% of $2000 = $60, so ATR stop ($40) is tighter → use ATR
        stop = rg.calculate_stop_loss_price(entry_price=2000.0, atr=20.0)
        assert stop == pytest.approx(1960.0)

        # Entry at $2000, ATR = $40 → stop distance = $80 (2x ATR)
        # 3% of $2000 = $60, so 3% cap ($60) is tighter → use cap
        stop = rg.calculate_stop_loss_price(entry_price=2000.0, atr=40.0)
        assert stop == pytest.approx(1940.0)

        # Check trigger: ATR=20 → stop at $1960
        assert rg.check_stop_loss_dynamic(entry_price=2000.0, current_price=1955.0, atr=20.0)  # Below stop
        assert not rg.check_stop_loss_dynamic(entry_price=2000.0, current_price=1965.0, atr=20.0)  # Above stop


class TestTradeDecisionLogic:
    """Tests for trade decision logic without full execution."""

    @pytest.mark.asyncio
    async def test_entry_conditions_direction(self):
        """Test entry requires UP direction."""
        executor = TradeExecutorService()

        # DOWN direction should not enter (now returns 3 values)
        action, reason, position_size = await executor._evaluate_entry(
            direction="DOWN",
            confidence=0.70,
            price=2000.0,
        )
        assert action == TradeAction.HOLD
        assert "DOWN" in reason
        assert position_size == 0.0

    @pytest.mark.asyncio
    async def test_entry_conditions_confidence(self):
        """Test entry requires sufficient confidence."""
        executor = TradeExecutorService()

        # Low confidence should not enter (now returns 3 values)
        action, reason, position_size = await executor._evaluate_entry(
            direction="UP",
            confidence=0.50,  # Below 0.60 threshold
            price=2000.0,
        )
        assert action == TradeAction.HOLD
        assert "Confidence" in reason
        assert position_size == 0.0


class TestEndToEndFlow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_prediction_to_hold_flow(self):
        """Test prediction with DOWN direction results in HOLD."""
        # This test simulates receiving a prediction event
        executor = TradeExecutorService()

        prediction = {
            "timestamp": 1234567890,
            "price": 2000.0,
            "direction": "DOWN",
            "confidence": 0.65,
            "shap_explanation": {},
        }

        # _evaluate_action now returns 3 values: (action, reason, position_size)
        action, reason, position_size = await executor._evaluate_action(
            direction=prediction["direction"],
            confidence=prediction["confidence"],
            price=prediction["price"],
        )

        assert action == TradeAction.HOLD
        assert "DOWN" in reason
        assert position_size == 0.0


# Run with: pytest tests/test_trade_executor.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
