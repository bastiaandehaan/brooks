#!/usr/bin/env python3
"""
Refactored FTMO Protection System Tests
"""

import os
import sys

import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.ftmo_guardian import FTMOGuardian, FTMORules
from execution.ftmo_state import FTMOState
from execution.trade_gate import check_ftmo_trade_gate


@pytest.fixture
def ftmo_rules_10k():
    """Fixture for 10k challenge rules"""
    return FTMORules.for_10k_challenge()


@pytest.fixture
def guardian(ftmo_rules_10k):
    """Fixture for FTMOGuardian"""
    return FTMOGuardian(rules=ftmo_rules_10k)


@pytest.fixture
def initial_equity():
    return 10000.0


@pytest.fixture
def state(initial_equity):
    """Fixture for FTMOState"""
    return FTMOState.initialize(initial_equity, day_tz="America/New_York")


def test_day_reset(state, initial_equity):
    """Test that day reset works correctly"""
    # Same day - no reset
    ts1 = pd.Timestamp("2026-01-23 14:00:00", tz="UTC")
    reset1 = state.update(10050.0, ts1)

    assert not reset1, "Should not reset on same day"
    assert abs(state.get_daily_pnl(10050.0) - 50.0) < 0.01, "Daily PnL should be $50"

    # Next day - should reset
    ts2 = pd.Timestamp("2026-01-24 14:00:00", tz="UTC")
    reset2 = state.update(10050.0, ts2)

    assert reset2, "Should reset on new day"
    assert abs(state.get_daily_pnl(10050.0)) < 0.01, "Daily PnL should reset to 0"


def test_trade_gate_blocking_normal(state, guardian, initial_equity):
    """Test that trade gate allows normal trading"""
    ts = pd.Timestamp("2026-01-23 14:00:00", tz="UTC")
    equity_normal = 10050.0
    state.update(equity_normal, ts)

    result = check_ftmo_trade_gate(
        equity_now=equity_normal,
        ftmo_state=state,
        ftmo_guardian=guardian,
        requested_risk_usd=100.0,
    )

    assert result.allowed
    assert result.capped_risk_usd == 100.0


def test_trade_gate_blocking_near_limit(guardian, initial_equity):
    """Test that trade gate caps risk near daily limits"""
    # Rules: max_daily_loss=5% ($500), buffer=1% ($100) -> safe limit = $400
    state = FTMOState.initialize(initial_equity)
    ts = pd.Timestamp("2026-01-23 14:00:00", tz="UTC")

    equity_loss = 9650.0  # -$350 daily loss (approaching $400 safe limit)
    state.update(equity_loss, ts)

    # Headroom = 400 - 350 = 50
    # Requesting 40 risk should be allowed because 350 + 40 = 390 <= 400
    result = check_ftmo_trade_gate(
        equity_now=equity_loss,
        ftmo_state=state,
        ftmo_guardian=guardian,
        requested_risk_usd=40.0,
    )

    assert result.allowed
    assert result.capped_risk_usd == 40.0


def test_trade_gate_blocking_over_limit(guardian, initial_equity):
    """Test that trade gate blocks trade over daily limit"""
    state = FTMOState.initialize(initial_equity)
    ts = pd.Timestamp("2026-01-23 14:00:00", tz="UTC")

    equity_over = 9550.0  # -$450 daily loss (over $400 safe limit)
    state.update(equity_over, ts)

    result = check_ftmo_trade_gate(
        equity_now=equity_over,
        ftmo_state=state,
        ftmo_guardian=guardian,
        requested_risk_usd=10.0,
    )

    assert not result.allowed
    assert "DAILY LOSS LIMIT" in result.reason


def test_risk_capping(state, guardian):
    """
    Test that risk capping works when can_trade passes but risk is slightly adjusted
    or headroom is checked.
    Note: Currently check_ftmo_trade_gate blocks if requested_risk exceeds headroom.
    """
    ts = pd.Timestamp("2026-01-23 14:00:00", tz="UTC")
    # Headroom = $400
    equity = 10000.0
    state.update(equity, ts)

    # Requesting $500 risk (exceeds $400 safe limit)
    result = check_ftmo_trade_gate(
        equity_now=equity,
        ftmo_state=state,
        ftmo_guardian=guardian,
        requested_risk_usd=500.0,
    )

    # Should be blocked because 0 + 500 > 400
    assert not result.allowed
    assert "DAILY LOSS LIMIT" in result.reason

    # Requesting $300 risk (within $400 safe limit)
    result2 = check_ftmo_trade_gate(
        equity_now=equity,
        ftmo_state=state,
        ftmo_guardian=guardian,
        requested_risk_usd=300.0,
    )

    assert result2.allowed
    assert result2.capped_risk_usd == 300.0


def test_pass_mode_trigger(guardian):
    """Test that pass mode triggers after profit target"""
    # Before target (10% = $1000 profit)
    status_before = guardian.get_account_status(10400.0, 0.0)
    assert status_before["total_profit"] < status_before["profit_target"]

    # After target
    status_after = guardian.get_account_status(11100.0, 0.0)
    assert status_after["total_profit"] >= status_after["profit_target"]


if __name__ == "__main__":
    pytest.main([__file__])
