#!/usr/bin/env python3
"""
MASTER TEST RUNNER - Run ALL tests in correct order
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and report status"""
    print("\n" + "=" * 80)
    print(f"  {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} FAILED (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"\n❌ {description} CRASHED: {e}")
        return False


def main():
    print("\n" + "🧪" * 40)
    print("  MASTER TEST RUNNER - Complete Test Suite")
    print("🧪" * 40)

    root = Path(__file__).parent.parent
    os.chdir(root)

    tests = [
        (
            ["python", "scripts/test_framework.py"],
            "1. Framework Integration Tests (MT5 + Components)",
        ),
        (
            ["pytest", "tests/test_selection_stability.py", "-v"],
            "2. Selection Stability Tests (Determinism)",
        ),
        (
            ["pytest", "tests/test_riskmanager.py", "-v"],
            "3. Risk Manager Tests (Sizing)",
        ),
        (
            ["pytest", "tests/test_h2l2.py", "-v"],
            "4. H2/L2 Strategy Tests",
        ),
        (
            ["pytest", "tests/test_context.py", "-v"],
            "5. Trend Detection Tests",
        ),
        (
            ["pytest", "tests/test_guardrails.py", "-v"],
            "6. Guardrails Tests (Session/Limits)",
        ),
        (
            ["pytest", "tests/", "-v", "--tb=short"],
            "7. Full Test Suite (All Tests)",
        ),
    ]

    results = []

    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))

        # Stop on first failure for faster feedback
        if not success:
            print("\n⚠️  Stopping at first failure for debugging")
            break

    # Summary
    print("\n" + "=" * 80)
    print("  COMPLETE TEST SUMMARY")
    print("=" * 80)

    for desc, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {desc}")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    print("\n" + "=" * 80)
    print(f"  FINAL SCORE: {passed}/{total} test suites passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 SUCCESS! All components working.")
        print("\n📋 NEXT STEPS:")
        print("  ✅ Framework verified")
        print("  ▶️  Run backtest: python -m backtest.runner --days 60")
        print("  ▶️  Optimize: python scripts/strategy_grid_search.py")
        print("  ▶️  Go live: python scripts/live_monitor.py")
        return 0
    else:
        print("\n❌ TESTS FAILED - Review output above")

        print("\nDebugging tips:")
        print("  • Check MT5 is running and logged in")
        print("  • Verify US500.cash symbol is available")
        print("  • Check timezone settings (should be UTC)")
        return 1
    if name == "main":
        sys.exit(main())
