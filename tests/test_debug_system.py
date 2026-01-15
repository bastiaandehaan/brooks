#!/usr/bin/env python3
"""
Test Debug Logging System
Verifies all logging functionality works correctly
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.debug_logger import DebugLogger, capture_error_context
from datetime import datetime
import pandas as pd


def test_error_logging():
    """Test error logging with full context"""
    print("\n" + "=" * 60)
    print("TEST 1: Error Logging")
    print("=" * 60)

    debug = DebugLogger(log_dir="logs_test")

    try:
        # Simulate an error
        raise ConnectionError("MT5 connection lost during signal check")
    except Exception as e:
        # Create fake market data
        fake_data = pd.DataFrame({
            'close': [5845, 5847, 5850, 5848, 5846],
        })

        context = capture_error_context(
            error=e,
            market_data=fake_data,
            config={
                "symbol": "US500.cash",
                "risk_pct": 0.5,
                "regime_filter": True,
                "chop_threshold": 2.0,
            },
            system_state={
                "balance": 10000,
                "daily_pnl": -150,
                "trades_today": 3,
                "time": datetime.now().isoformat(),
            }
        )

        error_path = debug.log_error(context)

        # Verify files created
        assert error_path.exists(), "JSON error log not created"
        assert error_path.with_suffix('.txt').exists(), "TXT error log not created"

        print(f"✅ Error log created: {error_path}")
        print(f"✅ Readable report: {error_path.with_suffix('.txt')}")

        # Show first few lines of readable report
        with open(error_path.with_suffix('.txt')) as f:
            lines = f.readlines()[:15]
        print("\n📄 Report preview:")
        print("".join(lines))

    return True


def test_trade_logging():
    """Test trade logging"""
    print("\n" + "=" * 60)
    print("TEST 2: Trade Logging")
    print("=" * 60)

    debug = DebugLogger(log_dir="logs_test")

    # Log 3 test trades
    trades = [
        {
            "timestamp": datetime.now().isoformat(),
            "side": "LONG",
            "entry": 5847.50,
            "stop": 5845.00,
            "tp": 5852.50,
            "result": "+2.0R",
            "pnl": 100.00,
        },
        {
            "timestamp": datetime.now().isoformat(),
            "side": "SHORT",
            "entry": 5850.00,
            "stop": 5852.00,
            "tp": 5846.00,
            "result": "-1.0R",
            "pnl": -50.00,
        },
        {
            "timestamp": datetime.now().isoformat(),
            "side": "LONG",
            "entry": 5845.00,
            "stop": 5843.00,
            "tp": 5849.00,
            "result": "+2.0R",
            "pnl": 100.00,
        }
    ]

    for trade in trades:
        debug.log_trade(trade)

    # Verify trades logged
    date_str = datetime.now().strftime("%Y%m%d")
    trade_file = Path("logs_test") / "trades" / f"trades_{date_str}.jsonl"

    assert trade_file.exists(), "Trade log file not created"

    # Count lines
    with open(trade_file) as f:
        lines = f.readlines()

    print(f"✅ Logged {len(lines)} trades to {trade_file}")
    print(f"✅ Trade log format: JSONL (one trade per line)")

    # Show first trade
    print("\n📊 First trade:")
    print(lines[0])

    return True


def test_snapshot():
    """Test system snapshot"""
    print("\n" + "=" * 60)
    print("TEST 3: System Snapshot")
    print("=" * 60)

    debug = DebugLogger(log_dir="logs_test")

    # Create fake data
    fake_market_data = pd.DataFrame({
        'open': [5840, 5842, 5845],
        'high': [5842, 5847, 5850],
        'low': [5838, 5841, 5844],
        'close': [5841, 5846, 5848],
    }, index=pd.date_range('2026-01-13 14:00', periods=3, freq='5min'))

    fake_trades = [
        {"side": "LONG", "entry": 5845, "result": "+2R"},
        {"side": "SHORT", "entry": 5850, "result": "-1R"},
    ]

    fake_account = {
        "balance": 10000,
        "equity": 10150,
        "margin_free": 9000,
    }

    snapshot_dir = debug.save_snapshot(
        market_data=fake_market_data,
        trades=fake_trades,
        account_info=fake_account
    )

    # Verify snapshot created
    assert snapshot_dir.exists(), "Snapshot directory not created"
    assert (snapshot_dir / "snapshot.json").exists(), "Snapshot metadata not created"
    assert (snapshot_dir / "market_data.csv").exists(), "Market data not saved"
    assert (snapshot_dir / "trades.json").exists(), "Trades not saved"
    assert (snapshot_dir / "account_info.json").exists(), "Account info not saved"

    print(f"✅ Snapshot created: {snapshot_dir}")
    print(f"✅ Contains: snapshot.json, market_data.csv, trades.json, account_info.json")

    return True


def test_daily_summary():
    """Test daily summary"""
    print("\n" + "=" * 60)
    print("TEST 4: Daily Summary")
    print("=" * 60)

    debug = DebugLogger(log_dir="logs_test")

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "trades": 5,
        "net_r": "+3.5R",
        "pnl_usd": 175.00,
        "winrate": 0.60,
        "max_dd": -1.0,
        "daily_sharpe": 2.1,
    }

    debug.save_daily_summary(summary)

    # Verify summary created
    date_str = datetime.now().strftime("%Y%m%d")
    summary_file = Path("logs_test") / "snapshots" / f"daily_summary_{date_str}.json"

    assert summary_file.exists(), "Daily summary not created"

    print(f"✅ Daily summary created: {summary_file}")

    # Show content
    import json
    with open(summary_file) as f:
        content = json.load(f)

    print("\n📊 Summary content:")
    print(json.dumps(content, indent=2))

    return True


def test_log_retrieval():
    """Test retrieving logs"""
    print("\n" + "=" * 60)
    print("TEST 5: Log Retrieval")
    print("=" * 60)

    debug = DebugLogger(log_dir="logs_test")

    # Get recent errors
    # recent_errors = debug.get_recent_logs(n=5) # Method missing
    print(f"✅ Found {len(recent_errors)} recent error logs")

    if recent_errors:
        print(f"   Most recent: {recent_errors[0]}")

    # Get today's trades
    date_str = datetime.now().strftime("%Y%m%d")
    trades = debug.get_daily_trades(date_str)
    print(f"✅ Found {len(trades)} trades for today")

    if trades:
        print(f"   First trade: {trades[0]}")

    return True


def cleanup_test_logs():
    """Clean up test logs"""
    import shutil

    test_dir = Path("logs_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n✅ Test logs cleaned up")


def main():
    print("\n" + "🧪" * 30)
    print("  DEBUG LOGGING SYSTEM - COMPREHENSIVE TEST")
    print("🧪" * 30)

    try:
        results = []

        # Run all tests
        results.append(("Error Logging", test_error_logging()))
        results.append(("Trade Logging", test_trade_logging()))
        results.append(("System Snapshot", test_snapshot()))
        results.append(("Daily Summary", test_daily_summary()))
        results.append(("Log Retrieval", test_log_retrieval()))

        # Summary
        print("\n" + "=" * 60)
        print("  TEST SUMMARY")
        print("=" * 60)

        all_passed = all(result[1] for result in results)

        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name:.<40} {status}")

        print("=" * 60)

        if all_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("\n📁 Test logs created in: logs_test/")
            print("   Review these files to see what will be generated during live trading")
            print("\n⚠️  Remember: Send files from logs/ folder when issues occur!")
        else:
            print("\n❌ SOME TESTS FAILED!")
            return 1

        # Ask user if they want to keep test logs
        print("\n" + "=" * 60)
        response = input("Delete test logs? (y/n): ")
        if response.lower() == 'y':
            cleanup_test_logs()
        else:
            print("✅ Test logs preserved in logs_test/ for inspection")

        return 0

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())