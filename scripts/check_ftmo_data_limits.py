# scripts/check_ftmo_data_limits.py
"""
Check EXACT data availability on FTMO Demo server
Find maximum reliable backtest period
"""

from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd


def check_data_depth():
    if not mt5.initialize():
        print("❌ MT5 init failed")
        return

    symbol = "US500.cash"

    print("=" * 80)
    print("FTMO DEMO DATA AVAILABILITY CHECK")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Server: {mt5.account_info().server}")
    print()

    # Test 1: Maximum single request (MT5 heeft vaak limiet ~100k bars)
    print("TEST 1: Single request limits")
    print("-" * 80)

    test_counts = [10000, 50000, 100000, 150000, 200000]
    max_working = 0

    for count in test_counts:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, count)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["datetime"] = pd.to_datetime(df["time"], unit="s")
            start = df["datetime"].iloc[0]
            end = df["datetime"].iloc[-1]
            days = (end - start).days

            print(f"✅ {count:7,d} bars → SUCCESS ({len(rates):7,d} bars, {days:4d} days)")
            print(f"   Range: {start.date()} to {end.date()}")
            max_working = count
        else:
            print(f"❌ {count:7,d} bars → FAILED (limit reached)")
            break

    print()
    print(f"Maximum single request: {max_working:,d} bars")

    # Test 2: How far back does data go?
    print()
    print("TEST 2: Historical depth")
    print("-" * 80)

    if max_working > 0:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, max_working)
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")

        oldest = df["datetime"].iloc[0]
        newest = df["datetime"].iloc[-1]
        total_days = (newest - oldest).days
        total_bars = len(df)

        print(f"Oldest available: {oldest}")
        print(f"Newest available: {newest}")
        print(f"Total span: {total_days} calendar days")
        print(f"Total bars: {total_bars:,d}")
        print(f"Avg bars/day: {total_bars / total_days:.1f}")

        # Calculate trading days (bars during session 09:30-16:00 ET)
        df["hour"] = df["datetime"].dt.hour
        session_bars = df[(df["hour"] >= 9) & (df["hour"] < 16)]
        trading_days = len(session_bars) / 26  # ~26 bars per trading day (6.5h × 4)

        print()
        print("RECOMMENDED BACKTEST PERIODS:")
        print("-" * 80)

        periods = [
            ("1 year", 252),
            ("2 years", 504),
            ("3 years", 756),
            ("Maximum available", int(trading_days)),
        ]

        for name, tdays in periods:
            if tdays <= trading_days:
                bars_needed = tdays * 26  # M15 bars
                cal_days = int(tdays * 365 / 252)  # Convert to calendar days

                if bars_needed <= total_bars:
                    # Calculate actual date range for this period
                    test_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars_needed)
                    if test_rates is not None:
                        test_df = pd.DataFrame(test_rates)
                        test_df["datetime"] = pd.to_datetime(test_df["time"], unit="s")
                        start = test_df["datetime"].iloc[0]
                        end = test_df["datetime"].iloc[-1]
                        actual_days = (end - start).days

                        print(
                            f"✅ {name:20s}: {tdays:3d} trading days ({actual_days:3d} calendar days)"
                        )
                        print(f"   Period: {start.date()} to {end.date()}")
                        print(f"   Bars: {bars_needed:,d} M15 bars")
                        print()

    print("=" * 80)
    print("RECOMMENDATION:")
    print("For robust backtesting, use MINIMUM 1 year (252 trading days)")
    print("For walk-forward validation, split data into train/validate/test")
    print("=" * 80)

    mt5.shutdown()


if __name__ == "__main__":
    check_data_depth()
