# backtest/optimizer.py
"""
FIXED: Config-first optimizer for Brooks strategy
Uses StrategyConfig defaults, varies specified parameters
EXPORTS best config as YAML for direct use in backtests
"""

import os
import sys
from datetime import datetime
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


class DummyArgs:
    """Minimal args object for run_backtest_from_config"""

    dashboard = "none"  # No dashboards during optimization (faster)
    export_risk_report = None


def start_optimization():
    """
    FIXED: Config-first parameter grid search

    Strategy:
    1. Start with default StrategyConfig
    2. Vary parameters in grid
    3. EXPORT best config as clean YAML
    """
    # Get repo root (parent of backtest/)
    script_dir = Path(__file__).parent  # backtest/
    repo_root = script_dir.parent  # brooks/

    print("\n" + "=" * 80)
    print("  BROOKS OPTIMIZER (CONFIG-FIRST)")
    print("=" * 80)
    print(f"  Repo root: {repo_root}")
    print("=" * 80)

    # Create base config with defaults
    base = StrategyConfig(
        symbol="US500.cash",
        regime_filter=True,
        regime_params=RegimeParams(
            chop_threshold=2.5,
            atr_period=14,
            range_period=20,
        ),
        trend_params=TrendParams(
            ema_period=20,
            min_slope=0.15,
        ),
        h2l2_params=H2L2Params(
            pullback_bars=3,
            signal_close_frac=0.30,
            min_risk_price_units=2.0,
            stop_buffer=1.0,
            cooldown_bars=20,
        ),
        guardrails=Guardrails(
            session_tz="America/New_York",
            day_tz="America/New_York",
            session_start="09:30",
            session_end="16:00",
            max_trades_per_day=2,
        ),
        risk_pct=1.0,
        costs_per_trade_r=0.04,
    )

    print(f"  Base config hash: {base.get_hash()}")
    print(f"  Symbol: {base.symbol}")
    print(f"  Regime filter: {base.regime_filter}")
    print(f"  Max trades/day: {base.guardrails.max_trades_per_day}")
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
    all_configs = []  # Store configs too
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
                args=DummyArgs(),  # Provide required args object
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
            all_configs.append(modified_config)  # Store the config

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
    best_idx = df_sorted.index[0]
    best = df_sorted.iloc[0]
    best_config = all_configs[best_idx]

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

    # --- EXPORT BEST CONFIG AS YAML ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure config directory exists
    config_dir = repo_root / "config" / "strategies"
    config_dir.mkdir(parents=True, exist_ok=True)

    output_yaml = config_dir / f"best_config_{timestamp}.yaml"

    print("\n" + "=" * 100)
    print("💾 EXPORTING BEST CONFIG")
    print("=" * 100)

    try:
        best_config.to_yaml(str(output_yaml))
        print(f"✅ Best config saved to: {output_yaml}")
        print(f"\n🚀 You can now run a backtest with:")
        print(
            f"   python backtest/runner.py --config {output_yaml.relative_to(repo_root)} \\"
        )
        print(f"          --start-date {START_DATE} --end-date {END_DATE}")
    except Exception as e:
        print(f"❌ Failed to save YAML: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 100)

    # --- COMPARISON WITH BASE CONFIG ---
    print("\n" + "=" * 100)
    print("📊 COMPARISON: Best Config vs Base Config")
    print("=" * 100)

    base_result = df[
        (df["max_trades_per_day"] == base.guardrails.max_trades_per_day)
        & (df["regime_filter"] == base.regime_filter)
        & (df["cooldown_bars"] == base.h2l2_params.cooldown_bars)
    ]

    if len(base_result) > 0:
        base_res = base_result.iloc[0]

        print("\nBASE CONFIG (defaults):")
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
        if base_res.get("net_r", 0) != 0:
            print(
                f"  Net R delta         : {delta_r:+.2f}R ({delta_r / base_res.get('net_r', 1) * 100:+.1f}%)"
            )
        else:
            print(f"  Net R delta         : {delta_r:+.2f}R")
        print(f"  Sharpe delta        : {delta_sharpe:+.3f}")
        print(f"  Trades delta        : {delta_trades:+d}")
    else:
        print("\n⚠️  Base config not in test grid")

    print("=" * 100)


if __name__ == "__main__":
    start_optimization()
