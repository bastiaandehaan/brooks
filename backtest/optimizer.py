# backtest/optimizer.py
"""
FIXED: Config-first optimizer for Brooks strategy
Uses StrategyConfig as base, only varies specified parameters
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import pandas as pd
from execution.guardrails import Guardrails
from strategies.config import StrategyConfig
from strategies.context import TrendParams
from strategies.h2l2 import H2L2Params
from strategies.regime import RegimeParams

# FIX: Import correct function
from backtest.runner import run_backtest_from_config


def start_optimization():
    """
    FIXED: Config-first parameter grid search

    Strategy:
    1. Load base config from YAML (single source of truth)
    2. Only vary parameters in grid
    3. Keep all other params from config
    """
    # --- BASE CONFIG ---
    # Get repo root (parent of backtest/)
    script_dir = Path(__file__).parent  # backtest/
    repo_root = script_dir.parent  # brooks/
    BASE_CONFIG = repo_root / "config" / "strategies" / "us500_sniper.yaml"

    # Convert to string for StrategyConfig.load()
    BASE_CONFIG = str(BASE_CONFIG)

    print("\n" + "=" * 80)
    print("  BROOKS OPTIMIZER (CONFIG-FIRST)")
    print("=" * 80)
    print(f"  Repo root: {repo_root}")
    print(f"  Base config: {BASE_CONFIG}")
    print(f"  Config exists: {Path(BASE_CONFIG).exists()}")
    print("=" * 80)

    # Load base config
    base = StrategyConfig.load(BASE_CONFIG)

    print(f"  Base config hash: {base.get_hash()}")
    print(f"  Symbol: {base.symbol}")
    print(f"  Regime filter: {base.regime_filter}")
    print("=" * 80)
    print()

    # --- TEST PERIOD ---
    # Use explicit date range (recommended over --days)
    START_DATE = "2024-01-24"
    END_DATE = "2026-01-24"

    print(f"📅 Test period: {START_DATE} to {END_DATE}")
    print()

    # --- THE PARAMETER GRID ---
    # Based on expert feedback from document 5:
    # Phase 1: Isolate edge source (frequency + regime)

    grid = {
        "max_trades_per_day": [1, 2],  # CRITICAL knob
        "regime_filter": [True, False],  # CRITICAL knob
        "cooldown_bars": [0, 20],  # Test Brooks-achtig (0) vs current (20)
    }

    # TOTAL: 2 × 2 × 2 = 8 combinations (fast, focused test)

    # Optional: Phase 2 grid (only if Phase 1 shows improvement)
    # Uncomment to test signal quality vs quantity:
    # grid.update({
    #     "signal_close_frac": [0.30, 0.35, 0.40],
    #     "min_risk_price_units": [2.0, 2.5],
    #     "chop_threshold": [2.5, 3.0],  # Less strict vs current
    # })

    # Make all combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_results = []
    total = len(combinations)

    print(f"🚀 Starting optimization")
    print(f"📊 Total combinations: {total}")
    print("-" * 80)

    for i, params in enumerate(combinations, 1):
        print(f"[{i}/{total}] Testing: {params}...", end="", flush=True)

        # Build modified config
        modified_config = StrategyConfig(
            symbol=base.symbol,
            # Regime (can be overridden)
            regime_filter=params.get("regime_filter", base.regime_filter),
            regime_params=RegimeParams(
                chop_threshold=params.get(
                    "chop_threshold", base.regime_params.chop_threshold
                ),
                atr_period=base.regime_params.atr_period,
                range_period=base.regime_params.range_period,
            ),
            # Trend (from base)
            trend_params=TrendParams(
                ema_period=params.get("ema_period", base.trend_params.ema_period),
                min_slope=params.get("min_slope", base.trend_params.min_slope),
            ),
            # H2/L2 (can be overridden)
            h2l2_params=H2L2Params(
                pullback_bars=params.get(
                    "pullback_bars", base.h2l2_params.pullback_bars
                ),
                signal_close_frac=params.get(
                    "signal_close_frac", base.h2l2_params.signal_close_frac
                ),
                min_risk_price_units=params.get(
                    "min_risk_price_units", base.h2l2_params.min_risk_price_units
                ),
                stop_buffer=params.get("stop_buffer", base.h2l2_params.stop_buffer),
                cooldown_bars=params.get(
                    "cooldown_bars", base.h2l2_params.cooldown_bars
                ),
            ),
            # Guardrails (can be overridden)
            guardrails=Guardrails(
                session_tz=base.guardrails.session_tz,
                day_tz=base.guardrails.day_tz,
                session_start=base.guardrails.session_start,
                session_end=base.guardrails.session_end,
                max_trades_per_day=params.get(
                    "max_trades_per_day", base.guardrails.max_trades_per_day
                ),
            ),
            # Risk/Costs (from base)
            risk_pct=base.risk_pct,
            costs_per_trade_r=base.costs_per_trade_r,
        )

        # Run backtest with FIXED function signature
        try:
            res = run_backtest_from_config(
                config=modified_config,
                start_date=START_DATE,
                end_date=END_DATE,
                initial_capital=10000.0,
                trading_days_per_year=252,
            )
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue

        # Check if successful
        if "error" not in res and res.get("trades", 0) > 0:
            row = {**params, **res}
            all_results.append(row)

            # Show key metrics
            sharpe = res.get("daily_sharpe_r", 0)
            net_r = res.get("net_r", 0)
            trades = res.get("trades", 0)
            wr = res.get("winrate", 0) * 100

            print(
                f" ✓ Sharpe={sharpe:.3f}, Net={net_r:+.1f}R, Trades={trades}, WR={wr:.1f}%"
            )
        else:
            error_msg = res.get("error", "Unknown error")
            print(f" ✗ {error_msg}")

    # --- ANALYSE ---
    if not all_results:
        print("\n❌ No results found")
        return

    df = pd.DataFrame(all_results)

    # Sort by Daily Sharpe (R-based)
    df_sorted = df.sort_values("daily_sharpe_r", ascending=False)
    top_5 = df_sorted.head(5)

    print("\n" + "=" * 100)
    print("🏆 TOP 5 CONFIGURATIONS (by Daily Sharpe R-based)")
    print("=" * 100)

    # Select columns to show
    cols_to_show = [
        "max_trades_per_day",
        "regime_filter",
        "cooldown_bars",
        "trades",
        "net_r",
        "daily_sharpe_r",
        "winrate",
        "profit_factor",
        "max_dd_r_daily",
    ]

    # Only show columns that exist
    cols_available = [c for c in cols_to_show if c in df.columns]
    print(top_5[cols_available].to_string(index=False))

    # Save full results
    output_file = "optimization_results.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\n✅ Full results saved: {output_file}")

    # Best config details
    best = top_5.iloc[0]
    print("\n" + "=" * 100)
    print("🎯 BEST CONFIGURATION:")
    print("=" * 100)

    # Show all parameters
    for key in grid.keys():
        if key in best:
            print(f"  {key:25s}: {best[key]}")

    print(f"\n  📊 PERFORMANCE:")
    print(f"  Trades              : {best.get('trades', 'N/A')}")
    print(f"  Net R               : {best.get('net_r', 0):+.2f}R")
    print(f"  Daily Sharpe (R/day): {best.get('daily_sharpe_r', 0):.3f}")
    print(f"  Winrate             : {best.get('winrate', 0) * 100:.1f}%")
    print(f"  Profit Factor       : {best.get('profit_factor', 0):.2f}")
    print(f"  Max DD (daily, R)   : {best.get('max_dd_r_daily', 0):.2f}R")
    print(f"  Annual R est.       : {best.get('annual_r', 0):+.2f}R")

    if best.get("regime_filter"):
        print(f"  Choppy segs skipped : {best.get('choppy_segments_skipped', 'N/A')}")

    print("=" * 100)

    # --- COMPARISON WITH BASE CONFIG ---
    print("\n" + "=" * 100)
    print("📄 COMPARISON: Best Config vs Base Config")
    print("=" * 100)

    base_result = df[
        (df["max_trades_per_day"] == base.guardrails.max_trades_per_day)
        & (df["regime_filter"] == base.regime_filter)
        & (df["cooldown_bars"] == base.h2l2_params.cooldown_bars)
    ]

    if len(base_result) > 0:
        base_res = base_result.iloc[0]

        print("\nBASE CONFIG (us500_sniper.yaml):")
        print(f"  Net R               : {base_res.get('net_r', 0):+.2f}R")
        print(f"  Daily Sharpe        : {base_res.get('daily_sharpe_r', 0):.3f}")
        print(f"  Trades              : {base_res.get('trades', 0)}")
        print(f"  Winrate             : {base_res.get('winrate', 0) * 100:.1f}%")

        print("\nBEST CONFIG (optimized):")
        print(f"  Net R               : {best.get('net_r', 0):+.2f}R")
        print(f"  Daily Sharpe        : {best.get('daily_sharpe_r', 0):.3f}")
        print(f"  Trades              : {best.get('trades', 0)}")
        print(f"  Winrate             : {best.get('winrate', 0) * 100:.1f}%")

        # Delta
        delta_r = best.get("net_r", 0) - base_res.get("net_r", 0)
        delta_sharpe = best.get("daily_sharpe_r", 0) - base_res.get("daily_sharpe_r", 0)
        delta_trades = best.get("trades", 0) - base_res.get("trades", 0)

        print(f"\nIMPROVEMENT:")
        print(
            f"  Net R delta         : {delta_r:+.2f}R ({delta_r / base_res.get('net_r', 1) * 100:+.1f}%)"
        )
        print(f"  Sharpe delta        : {delta_sharpe:+.3f}")
        print(f"  Trades delta        : {delta_trades:+d}")
    else:
        print("\n⚠️  Base config not in test grid")

    print("=" * 100)


if __name__ == "__main__":
    start_optimization()
