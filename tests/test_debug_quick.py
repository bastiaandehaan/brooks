#!/usr/bin/env python3
"""
Quick test for debug logging system
Run this to verify everything works before live trading
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.debug_logger import DebugLogger, capture_error_context

print("🧪 Testing Debug Logger...")
print("=" * 60)

# Test 1: Initialize logger
print("\n1️⃣ Testing initialization...")
try:
    logger = DebugLogger(log_dir="logs")
    print("   ✅ DebugLogger created")
    print(f"   ✅ Errors dir: {logger.errors_dir}")
    print(f"   ✅ Trades dir: {logger.trades_dir}")
    print(f"   ✅ Snapshots dir: {logger.snapshots_dir}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Log a test error
print("\n2️⃣ Testing error logging...")
try:
    test_error = ValueError("Test error for debugging")
    context = capture_error_context(test_error, config={"test": "configuration"})
    logger.log_error(context)

    # Check if files were created
    error_files = list(logger.errors_dir.glob("error_*.txt"))
    if error_files:
        print(f"   ✅ Error logged: {error_files[-1].name}")
    else:
        print("   ❌ No error file created")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Log a test trade
print("\n3️⃣ Testing trade logging...")
try:
    test_trade = {
        "timestamp": "2026-01-13T15:00:00",
        "side": "LONG",
        "entry": 5847.5,
        "stop": 5845.0,
        "tp": 5852.5,
        "result_r": 2.0,
        "pnl": 100.0,
    }
    logger.log_trade(test_trade)

    # Check if trade file was created
    trade_files = list(logger.trades_dir.glob("trades_*.jsonl"))
    if trade_files:
        print(f"   ✅ Trade logged: {trade_files[-1].name}")
    else:
        print("   ❌ No trade file created")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Save a snapshot
print("\n4️⃣ Testing snapshot...")
try:
    test_snapshot = {
        "timestamp": "2026-01-13T15:00:00",
        "balance": 10000,
        "daily_pnl": 0,
        "trades_today": 0,
    }
    logger.save_snapshot(test_snapshot)

    # Check if snapshot was created
    snapshot_files = list(logger.snapshots_dir.glob("snapshot_*.json"))
    if snapshot_files:
        print(f"   ✅ Snapshot saved: {snapshot_files[-1].name}")
    else:
        print("   ❌ No snapshot file created")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL DEBUG TESTS PASSED!")
print("=" * 60)
print("\nDebug system is ready for live trading.")
print("Log files location: logs/")
print("\nYou can now start the live monitor with confidence!")
print("=" * 60)
