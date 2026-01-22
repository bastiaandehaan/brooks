# scripts/test_ftmo_data_limits.py
"""
Complete test van FTMO data limieten voor US500.cash

Test alle timeframes en ontdek de exacte limieten:
- M1 (1 minuut)
- M5 (5 minuten)
- M15 (15 minuten)
- H1 (1 uur)
- D1 (dagelijks)

Voor elk timeframe:
1. Test met verschillende bar counts
2. Bepaal maximum dat FTMO vrijgeeft
3. Verifieer echte calendar coverage
"""

import os
import sys

import MetaTrader5 as mt5

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates


def test_timeframe_limit(
    mt5_client,
    symbol: str,
    timeframe: int,
    timeframe_name: str,
    bars_per_day: int,
    test_counts: list[int],
):
    """
    Test een specifiek timeframe met verschillende bar counts

    Returns:
        Dict met resultaten
    """
    print("\n" + "=" * 80)
    print(f"  TESTING {timeframe_name} TIMEFRAME")
    print("=" * 80)

    results = {"timeframe": timeframe_name, "bars_per_day": bars_per_day, "tests": []}

    for requested in test_counts:
        print(f"\n→ Requesting {requested:,} bars ({requested / bars_per_day:.1f} days)...")

        try:
            req = RatesRequest(symbol, timeframe, requested, pos=0)
            data = fetch_rates(mt5_client._mt5, req)

            if data.empty:
                print("   ❌ FAILED: Empty dataset")
                results["tests"].append(
                    {
                        "requested": requested,
                        "received": 0,
                        "success": False,
                        "error": "Empty dataset",
                    }
                )
                continue

            received = len(data)
            first_bar = data.index[0]
            last_bar = data.index[-1]
            calendar_days = (last_bar - first_bar).days

            success = received == requested
            status = "✅ OK" if success else f"⚠️  PARTIAL ({received / requested * 100:.1f}%)"

            print(f"   {status}")
            print(f"   Received   : {received:,} bars")
            print(f"   First bar  : {first_bar}")
            print(f"   Last bar   : {last_bar}")
            print(f"   Coverage   : {calendar_days} calendar days")
            print(f"   Trading days: ~{received / bars_per_day:.1f} days")

            results["tests"].append(
                {
                    "requested": requested,
                    "received": received,
                    "first_bar": str(first_bar),
                    "last_bar": str(last_bar),
                    "calendar_days": calendar_days,
                    "trading_days": received / bars_per_day,
                    "success": success,
                    "coverage_pct": (received / requested) * 100,
                }
            )

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results["tests"].append(
                {"requested": requested, "received": 0, "success": False, "error": str(e)}
            )

    return results


def find_exact_limit(
    mt5_client, symbol: str, timeframe: int, timeframe_name: str, bars_per_day: int
):
    """
    Binary search om exacte limiet te vinden
    """
    print("\n" + "🔍" * 40)
    print(f"  FINDING EXACT LIMIT FOR {timeframe_name}")
    print("🔍" * 40)

    # Start met breed bereik
    low = 50000
    high = 200000
    exact_limit = None

    while low <= high:
        mid = (low + high) // 2

        print(f"\n→ Testing {mid:,} bars...")

        try:
            req = RatesRequest(symbol, timeframe, mid, pos=0)
            data = fetch_rates(mt5_client._mt5, req)
            received = len(data)

            if received == mid:
                # Gelukt! Probeer hoger
                print(f"   ✅ Success: {received:,} bars")
                exact_limit = mid
                low = mid + 1000  # Probeer 1000 meer
            else:
                # Partial data - limiet bereikt
                print(f"   ⚠️  Partial: {received:,}/{mid:,} bars")
                high = mid - 1000
        except Exception as e:
            print(f"   ❌ Error: {e}")
            high = mid - 1000

    if exact_limit:
        print(
            f"\n🎯 EXACT LIMIT: {exact_limit:,} bars ({exact_limit / bars_per_day:.1f} trading days)"
        )
    else:
        print(f"\n⚠️  Could not determine exact limit (below {low:,} bars)")

    return exact_limit


def main():
    """Main test uitvoeren"""

    print("\n" + "🧪" * 40)
    print("  FTMO DATA LIMITS TEST - US500.cash")
    print("  Testing ALL timeframes to find exact limits")
    print("🧪" * 40)

    # Connect
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        print("❌ Failed to connect to MT5")
        return 1

    symbol = "US500.cash"

    # Test verschillende timeframes
    all_results = []

    # =====================================================
    # 1. M1 (1 minuut) - 1440 bars/dag
    # =====================================================
    print("\n" + "=" * 80)
    print("  📊 1. M1 (1 MINUTE) BARS")
    print("=" * 80)

    m1_results = test_timeframe_limit(
        client,
        symbol,
        mt5.TIMEFRAME_M1,
        "M1",
        bars_per_day=1440,
        test_counts=[
            1440,  # 1 dag
            1440 * 7,  # 1 week
            1440 * 30,  # 1 maand
            1440 * 60,  # 2 maanden
            1440 * 90,  # 3 maanden
        ],
    )
    all_results.append(m1_results)

    # Zoek exact limiet voor M1
    m1_limit = find_exact_limit(client, symbol, mt5.TIMEFRAME_M1, "M1", 1440)

    # =====================================================
    # 2. M5 (5 minuten) - 288 bars/dag
    # =====================================================
    print("\n" + "=" * 80)
    print("  📊 2. M5 (5 MINUTES) BARS")
    print("=" * 80)

    m5_results = test_timeframe_limit(
        client,
        symbol,
        mt5.TIMEFRAME_M5,
        "M5",
        bars_per_day=288,
        test_counts=[
            288 * 30,  # 1 maand
            288 * 90,  # 3 maanden
            288 * 180,  # 6 maanden
            288 * 340,  # 340 dagen (jouw test)
            288 * 365,  # 1 jaar
            288 * 500,  # 500 dagen
        ],
    )
    all_results.append(m5_results)

    m5_limit = find_exact_limit(client, symbol, mt5.TIMEFRAME_M5, "M5", 288)

    # =====================================================
    # 3. M15 (15 minuten) - 96 bars/dag
    # =====================================================
    print("\n" + "=" * 80)
    print("  📊 3. M15 (15 MINUTES) BARS")
    print("=" * 80)

    m15_results = test_timeframe_limit(
        client,
        symbol,
        mt5.TIMEFRAME_M15,
        "M15",
        bars_per_day=96,
        test_counts=[
            96 * 90,  # 3 maanden
            96 * 180,  # 6 maanden
            96 * 340,  # 340 dagen
            96 * 340 * 2,  # 680 dagen (jouw test!)
            96 * 365,  # 1 jaar
            96 * 365 * 2,  # 2 jaar
            96 * 365 * 3,  # 3 jaar
        ],
    )
    all_results.append(m15_results)

    m15_limit = find_exact_limit(client, symbol, mt5.TIMEFRAME_M15, "M15", 96)

    # =====================================================
    # 4. H1 (1 uur) - 24 bars/dag
    # =====================================================
    print("\n" + "=" * 80)
    print("  📊 4. H1 (1 HOUR) BARS")
    print("=" * 80)

    h1_results = test_timeframe_limit(
        client,
        symbol,
        mt5.TIMEFRAME_H1,
        "H1",
        bars_per_day=24,
        test_counts=[
            24 * 365,  # 1 jaar
            24 * 365 * 2,  # 2 jaar
            24 * 365 * 3,  # 3 jaar
            24 * 365 * 5,  # 5 jaar
        ],
    )
    all_results.append(h1_results)

    # =====================================================
    # SUMMARY REPORT
    # =====================================================
    print("\n" + "=" * 80)
    print("  📋 FTMO DATA LIMITS SUMMARY")
    print("=" * 80)

    print(f"\n{'Timeframe':<12} {'Max Bars':<15} {'Max Days':<15} {'Status':<10}")
    print("-" * 80)

    for result in all_results:
        tf = result["timeframe"]
        bars_per_day = result["bars_per_day"]

        # Find highest successful test
        successful = [t for t in result["tests"] if t["success"]]
        if successful:
            max_bars = max(t["received"] for t in successful)
            max_days = max_bars / bars_per_day
            status = "✅ OK"
        else:
            # Find highest partial
            partial = [t for t in result["tests"] if t.get("received", 0) > 0]
            if partial:
                max_bars = max(t["received"] for t in partial)
                max_days = max_bars / bars_per_day
                status = "⚠️  LIMITED"
            else:
                max_bars = 0
                max_days = 0
                status = "❌ FAILED"

        print(f"{tf:<12} {max_bars:<15,} {max_days:<15.1f} {status:<10}")

    # =====================================================
    # VERIFICATION: Wat gebruikt je backtest?
    # =====================================================
    print("\n" + "=" * 80)
    print("  🔍 VERIFICATION: Your Backtest Settings")
    print("=" * 80)

    days = 340
    m15_requested = days * 96 * 2  # 65,280
    m5_requested = days * 288  # 97,920

    print("\nYour backtest requests:")
    print(f"  M15: {m15_requested:,} bars ({m15_requested / 96:.1f} days)")
    print(f"  M5 : {m5_requested:,} bars ({m5_requested / 288:.1f} days)")

    # Check tegen limieten
    m15_success = any(t["success"] for t in m15_results["tests"] if t["requested"] >= m15_requested)
    m5_success = any(t["success"] for t in m5_results["tests"] if t["requested"] >= m5_requested)

    print("\nCan FTMO provide this?")
    print(f"  M15: {'✅ YES' if m15_success else '❌ NO - EXCEEDS LIMIT!'}")
    print(f"  M5 : {'✅ YES' if m5_success else '❌ NO - EXCEEDS LIMIT!'}")

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================
    print("\n" + "=" * 80)
    print("  💡 RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on FTMO limits:

1. M1 Data:
   - Typically limited to 30-90 days
   - Use ONLY for both-hit resolution
   - Don't rely on it for backtesting

2. M5 Data:  
   - Usually 300-500 days available
   - Perfect for your 340-day backtest ✅

3. M15 Data:
   - Usually 1-3 years available
   - Perfect for context (680 days) ✅

4. If you need MORE data:
   - Download CSV from broker
   - Use alternative data source
   - Or accept the limitation

5. Your current setup (340 days):
   - Should work fine if limits allow
   - Verify with this script's output above
""")

    client.shutdown()

    print("\n" + "=" * 80)
    print("  ✅ TEST COMPLETE")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
