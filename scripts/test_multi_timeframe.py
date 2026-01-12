# scripts/test_multi_timeframe.py
"""
Test strategy over 60, 180, and 340 days
Validates consistency and robustness
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import run_backtest
import pandas as pd


def main():
    print("\n" + "🎯" * 40)
    print("  BROOKS MULTI-TIMEFRAME VALIDATION")
    print("🎯" * 40 + "\n")

    timeframes = [60, 180, 340]
    results = []

    for days in timeframes:
        print(f"\n{'=' * 80}")
        print(f"  TESTING: {days} DAYS")
        print(f"{'=' * 80}\n")

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=0.15,
            ema_period=20,
            pullback_bars=3,
            signal_close_frac=0.30,
            stop_buffer=2.0,
            min_risk_price_units=2.0,
            cooldown_bars=10,
        )

        if "error" not in metrics:
            results.append(metrics)

    # Summary
    print("\n" + "=" * 80)
    print("  📊 MULTI-TIMEFRAME SUMMARY")
    print("=" * 80 + "\n")

    df = pd.DataFrame(results)

    print(df[[
        "days", "trades", "net_r", "winrate", "sharpe",
        "profit_factor", "max_dd"
    ]].to_string(index=False))

    print("\n" + "=" * 80)
    print("  🎯 ROBUSTNESS CHECK")
    print("=" * 80 + "\n")

    if len(results) >= 2:
        sharpe_60 = results[0]["sharpe"]
        sharpe_180 = results[1]["sharpe"]
        sharpe_340 = results[2]["sharpe"] if len(results) > 2 else None

        print(f"  Sharpe degradation 60→180d: {(sharpe_180 / sharpe_60 - 1) * 100:+.1f}%")
        if sharpe_340:
            print(f"  Sharpe degradation 180→340d: {(sharpe_340 / sharpe_180 - 1) * 100:+.1f}%")

        print()
        if sharpe_180 / sharpe_60 > 0.8:
            print("  ✅ ROBUST: Strategy holds up over time")
        else:
            print("  ⚠️  DEGRADATION: Strategy weakens over time (possible overfit)")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()