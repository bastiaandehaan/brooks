#!/usr/bin/env python3
"""
COMPLETE FRAMEWORK TEST SUITE

Run this FIRST to verify all components work.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from execution.risk_manager import RiskManager, RiskParams
from execution.selection import select_top_per_ny_day
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_h2l2_trades
from strategies.regime import RegimeParams, should_trade_today
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates


def test_mt5_connection():
    """Test 1: MT5 Connection"""
    print("\n" + "=" * 80)
    print("TEST 1: MT5 Connection")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        print("❌ MT5 connection failed")
        return False

    print("✅ MT5 connected")

    spec = client.get_symbol_specification("US500.cash")
    if spec is None:
        print("❌ Symbol spec failed")
        client.shutdown()
        return False

    print(f"✅ Symbol spec: {spec.name}, contract={spec.contract_size}")
    client.shutdown()
    return True


def test_data_fetch():
    """Test 2: Data Fetching"""
    print("\n" + "=" * 80)
    print("TEST 2: Data Fetching")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        m15 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M15, 100))
        m5 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M5, 100))

        if m15.empty or m5.empty:
            print("❌ Empty dataframes")
            return False

        print(f"✅ M15: {len(m15)} bars")
        print(f"✅ M5: {len(m5)} bars")
        print(f"✅ Columns: {list(m5.columns)}")
        return True

    finally:
        client.shutdown()


def test_regime_filter():
    """Test 3: Regime Detection"""
    print("\n" + "=" * 80)
    print("TEST 3: Regime Detection")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        m15 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M15, 500))

        params = RegimeParams(chop_threshold=2.5)
        should_trade, reason = should_trade_today(m15, params)

        print(f"✅ Regime check: {reason}")
        print(f"✅ Should trade: {should_trade}")
        return True

    finally:
        client.shutdown()


def test_trend_detection():
    """Test 4: Trend Detection"""
    print("\n" + "=" * 80)
    print("TEST 4: Trend Detection")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        m15 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M15, 300))

        params = TrendParams(min_slope=0.10, ema_period=20)
        trend, metrics = infer_trend_m15(m15, params)

        print(f"✅ Trend: {trend}")
        print(f"✅ EMA: {metrics.last_ema:.2f}")
        print(f"✅ Slope: {metrics.ema_slope:.4f}")
        return True

    finally:
        client.shutdown()


def test_h2l2_planning():
    """Test 5: H2/L2 Trade Planning"""
    print("\n" + "=" * 80)
    print("TEST 5: H2/L2 Planning")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        spec = client.get_symbol_specification("US500.cash")
        m5 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M5, 500))

        params = H2L2Params(
            pullback_bars=3,
            signal_close_frac=0.30,
            min_risk_price_units=1.0,
            stop_buffer=1.0,
        )

        trades = plan_h2l2_trades(m5, Side.LONG, spec, params)

        print(f"✅ Planned trades: {len(trades)}")
        if trades:
            print(f"✅ First trade: {trades[0].reason}")
        return True

    finally:
        client.shutdown()


def test_guardrails():
    """Test 6: Guardrails"""
    print("\n" + "=" * 80)
    print("TEST 6: Guardrails")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        spec = client.get_symbol_specification("US500.cash")
        m5 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M5, 500))

        trades = plan_h2l2_trades(
            m5,
            Side.LONG,
            spec,
            H2L2Params(pullback_bars=3, signal_close_frac=0.30, min_risk_price_units=1.0),
        )

        g = Guardrails(
            session_tz="America/New_York",
            session_start="09:30",
            session_end="16:00",
            max_trades_per_day=2,
        )

        accepted, rejected = apply_guardrails(trades, g)

        print(f"✅ Candidates: {len(trades)}")
        print(f"✅ Accepted: {len(accepted)}")
        print(f"✅ Rejected: {len(rejected)}")
        return True

    finally:
        client.shutdown()


def test_risk_sizing():
    """Test 7: Risk Sizing"""
    print("\n" + "=" * 80)
    print("TEST 7: Risk Sizing")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        spec = client.get_symbol_specification("US500.cash")

        rm = RiskManager(RiskParams(min_risk_pts=1.0))

        lots, risk_usd = rm.size_position(
            balance=10000.0,
            entry=5850.0,
            stop=5840.0,  # 10 points risk
            risk_pct=0.5,  # 0.5%
            spec=spec,
        )

        print(f"✅ Lots: {lots:.2f}")
        print(f"✅ Risk USD: ${risk_usd:.2f}")
        print(f"✅ Expected ~$50 (0.5% of $10k)")

        assert 45 < risk_usd < 55, "Risk calculation wrong!"
        return True

    finally:
        client.shutdown()


def test_selection():
    """Test 8: Daily Selection"""
    print("\n" + "=" * 80)
    print("TEST 8: Daily Selection")
    print("=" * 80)

    client = Mt5Client(mt5_module=mt5)
    client.initialize()

    try:
        spec = client.get_symbol_specification("US500.cash")
        m5 = fetch_rates(mt5, RatesRequest("US500.cash", mt5.TIMEFRAME_M5, 1000))

        trades = plan_h2l2_trades(
            m5,
            Side.LONG,
            spec,
            H2L2Params(pullback_bars=3, signal_close_frac=0.30, min_risk_price_units=1.0),
        )

        selected, stats = select_top_per_ny_day(
            trades,
            max_trades_day=2,
            tick_size=spec.tick_size,
        )

        print(f"✅ Candidates: {len(trades)}")
        print(f"✅ Selected: {len(selected)}")
        print(f"✅ Days: {len(stats)}")
        return True

    finally:
        client.shutdown()


def main():
    print("\n" + "🧪" * 40)
    print("  BROOKS FRAMEWORK - COMPLETE TEST SUITE")
    print("🧪" * 40)

    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("Data Fetch", test_data_fetch),
        ("Regime Filter", test_regime_filter),
        ("Trend Detection", test_trend_detection),
        ("H2/L2 Planning", test_h2l2_planning),
        ("Guardrails", test_guardrails),
        ("Risk Sizing", test_risk_sizing),
        ("Daily Selection", test_selection),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} CRASHED: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:.<50} {status}")

    passed_count = sum(1 for _, p in results if p)
    total = len(results)

    print("=" * 80)
    print(f"\n  RESULT: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n🎉 ALL TESTS PASSED! Framework is ready.")
        print("\nNext steps:")
        print("  1. pytest -q  (run full test suite)")
        print("  2. python -m backtest.runner --days 60")
        print("  3. python scripts/strategy_grid_search.py")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
