"""
FTMO Income Optimizer
--------------------
Purpose:
- Systematically optimize Brooks-style strategy parameters
- Target FTMO survivability + monthly income (not Sharpe / hedge fund metrics)

Usage:
python -m backtest.optimizer --base-config path/to/base.yaml --days 1500
"""

from __future__ import annotations

import argparse
import itertools
from copy import deepcopy
from pathlib import Path

import pandas as pd
from strategies.config import StrategyConfig

from backtest.runner import run_backtest_from_config

# ============================================================
# PARAMETER SEARCH SPACE
# ============================================================

SEARCH_SPACE = {
    "risk_pct": [0.6, 0.7, 0.8, 0.9, 1.0],
    "h2l2_params.pullback_bars": [2, 3, 4],
    "h2l2_params.signal_close_frac": [0.25, 0.35, 0.45],
    "trend_params.ema_period": [13, 15, 20],
    "trend_params.min_slope": [0.08, 0.10, 0.12],
    "regime_params.chop_threshold": [2.5, 3.0, 3.5],
    "guardrails.max_trades_per_day": [1, 2],
}


# ============================================================
# UTILITIES
# ============================================================


def set_nested(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def generate_param_combos():
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ============================================================
# SCORING FUNCTION (FTMO-FOCUSED)
# ============================================================


def compute_score(net_r: float, max_dd: float, breach_prob: float) -> float:
    """
    Higher is better.

    - Net R rewarded
    - Drawdown penalized
    - FTMO breach probability heavily penalized
    """
    return net_r - abs(max_dd) * 2.0 - breach_prob * 200.0


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser("FTMO income optimizer")

    parser.add_argument("--base-config", required=True, type=str)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--trading-days-year", type=int, default=252)
    parser.add_argument("--csv-out", type=str, default="optimizer_results.csv")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Resolve base config safely
    # ------------------------------------------------------------

    base_path = Path(args.base_config).expanduser().resolve()

    if not base_path.exists():
        raise ValueError(f"Base config not found: {base_path}")

    if base_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Base config must be a YAML file")

    base_cfg = StrategyConfig.load(base_path)

    print("=" * 80)
    print("📋 LOADED STRATEGY CONFIG")
    print("=" * 80)
    print(f"  File: {base_path}")
    # print(f"  Hash: {base_cfg.hash}")  # StrategyConfig has no .hash in this repo
    print(f"  Symbol: {base_cfg.symbol}")
    print(f"  Regime Filter: {base_cfg.regime_filter}")
    print(f"  Risk per Trade: {base_cfg.risk_pct}%")
    print(f"  Max Trades/Day: {base_cfg.guardrails.max_trades_per_day}")
    print("=" * 80)

    combos = list(generate_param_combos())
    print(f"Optimizer combos: {len(combos)}")

    results = []

    # ------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------

    for params in combos:
        cfg_dict = deepcopy(base_cfg.to_dict())

        for k, v in params.items():
            set_nested(cfg_dict, k, v)

        cfg = StrategyConfig.from_dict(cfg_dict)

        bt = run_backtest_from_config(
            cfg,
            days=args.days,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            trading_days_year=args.trading_days_year,
            dashboard="none",
        )

        if bt is None or bt.trades.empty:
            continue

        net_r = bt.metrics.net_r
        max_dd = bt.metrics.max_drawdown_r
        breach_prob = bt.risk.total_breach_probability

        score = compute_score(net_r, max_dd, breach_prob)

        results.append(
            {
                "score": score,
                "net_r": net_r,
                "max_dd": max_dd,
                "breach_prob": breach_prob,
                "trades": len(bt.trades),
                "avg_r": bt.metrics.avg_r,
                "winrate": bt.metrics.winrate,
                **params,
            }
        )

    # ------------------------------------------------------------
    # Aggregate & export
    # ------------------------------------------------------------

    if not results:
        raise RuntimeError("Optimizer produced no valid results")

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    df.to_csv(args.csv_out, index=False)

    print("=" * 80)
    print("🏁 OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(df.head(5))
    print(f"\nSaved results to: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
