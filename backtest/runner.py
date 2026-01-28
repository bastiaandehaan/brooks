# backtest/runner.py
"""
Brooks Backtest Runner - CLEANED & FIXED VERSION

Usage:
    python -m backtest.runner --config config/strategies/us500_final.yaml --start-date 2024-01-24 --end-date 2026-01-24
    python -m backtest.runner --config config/strategies/us500_final.yaml --days 340
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execution.guardrails import Guardrails, apply_guardrails
from execution.selection import select_top_per_ny_day
from strategies.config import StrategyConfig
from strategies.context import Trend, TrendParams, infer_trend_m15_series
from strategies.h2l2 import H2L2Params, PlannedTrade, Side, plan_h2l2_trades
from strategies.regime import MarketRegime, detect_regime_series
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates

# Suppress noisy logging
logging.getLogger("execution.guardrails").setLevel(logging.WARNING)
logging.getLogger("execution.selection").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

NY_TZ = "America/New_York"


def _normalize_ohlc(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")
    out = df.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    return out


def _trend_to_side(trend: Trend) -> Side | None:
    if trend == Trend.BULL:
        return Side.LONG
    if trend == Trend.BEAR:
        return Side.SHORT
    return None


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> tuple[float, pd.Timestamp]:
    """Simulate trade with worst-case both-hit policy."""
    future = m5_data.loc[t.execute_ts :]
    last_ts = pd.to_datetime(future.index[-1]) if len(future) else pd.to_datetime(t.execute_ts)

    for ts, bar in future.iterrows():
        high, low = float(bar["high"]), float(bar["low"])
        ts = pd.to_datetime(ts)

        if t.side == Side.LONG:
            hit_sl, hit_tp = low <= t.stop, high >= t.tp
            if hit_sl and hit_tp:
                return -1.0, ts
            if hit_sl:
                return -1.0, ts
            if hit_tp:
                return 2.0, ts
        else:
            hit_sl, hit_tp = high >= t.stop, low <= t.tp
            if hit_sl and hit_tp:
                return -1.0, ts
            if hit_sl:
                return -1.0, ts
            if hit_tp:
                return 2.0, ts

    return 0.0, last_ts


def _build_trades_dataframe(
    final_trades: List[PlannedTrade],
    results_r: List[float],
    exit_ts_list: List[pd.Timestamp],
    m5_data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for trade, result, exit_ts in zip(final_trades, results_r, exit_ts_list, strict=True):
        entry_ts = pd.to_datetime(trade.execute_ts)
        exit_ts = pd.to_datetime(exit_ts)

        regime_val = None
        try:
            row = m5_data.loc[trade.execute_ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            regime_val = row.get("regime", None)
        except Exception:
            pass

        rows.append(
            {
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "side": trade.side.value,
                "entry": float(trade.entry),
                "stop": float(trade.stop),
                "tp": float(trade.tp),
                "net_r": float(result),
                "reason": str(trade.reason),
                "regime_at_entry": str(regime_val) if regime_val else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    if df["exit_time"].dt.tz is None:
        df["ny_day"] = df["exit_time"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ).dt.date
    else:
        df["ny_day"] = df["exit_time"].dt.tz_convert(NY_TZ).dt.date

    return df


def _daily_series_from_trades(
    trades_df: pd.DataFrame,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype="float64")

    daily = trades_df.groupby("ny_day")["net_r"].sum().astype("float64")

    if start_dt is not None and end_dt is not None:
        s = (
            pd.to_datetime(start_dt).tz_convert(NY_TZ)
            if start_dt.tzinfo
            else pd.to_datetime(start_dt).tz_localize("UTC").tz_convert(NY_TZ)
        )
        e = (
            pd.to_datetime(end_dt).tz_convert(NY_TZ)
            if end_dt.tzinfo
            else pd.to_datetime(end_dt).tz_localize("UTC").tz_convert(NY_TZ)
        )
        full_days = pd.date_range(s.date(), e.date(), freq="D").date
        daily = daily.reindex(full_days, fill_value=0.0)

    return daily


def _compute_metrics(
    trades_df: pd.DataFrame,
    daily_pnl_r: pd.Series,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if daily_pnl_r.empty:
        return out

    dp = pd.Series(daily_pnl_r, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    mu = float(dp.mean())
    sig = float(dp.std(ddof=1)) if len(dp) > 1 else 0.0
    daily_sharpe_r = (mu / sig) * np.sqrt(trading_days_per_year) if sig > 0 else 0.0

    eq_r = dp.cumsum()
    run_max = eq_r.cummax()
    dd_r = eq_r - run_max
    max_dd_r_daily = float(dd_r.min())

    out.update(
        {
            "daily_sharpe_r": daily_sharpe_r,
            "max_dd_r_daily": max_dd_r_daily,
            "best_day_r": float(dp.max()),
            "worst_day_r": float(dp.min()),
            "pct_pos_days": float((dp > 0).mean() * 100.0) if len(dp) else 0.0,
            "calendar_days": int(len(dp)),
            "days_with_trades": int((dp != 0).sum()),
        }
    )

    return out


def run_backtest_from_config(
    config: StrategyConfig,
    *,
    days: int = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 10000.0,
    trading_days_per_year: int = 252,
    dashboard: str = "v2",
) -> Dict[str, Any]:
    """
    Run backtest using StrategyConfig.

    Args:
        config: Strategy configuration
        days: Number of days (legacy mode)
        start_date: Start date YYYY-MM-DD (preferred)
        end_date: End date YYYY-MM-DD (preferred)
        initial_capital: Starting capital
        trading_days_per_year: For Sharpe calculation
        dashboard: "v2", "v1", or "none"
    """
    symbol = config.symbol

    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        return {"error": "MT5 init failed"}

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return {"error": "Symbol not found"}

    # Determine backtest mode
    if start_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")
        cal_days = (end_dt - start_dt).days
        count_m15 = max(cal_days * 90, 10000)
        count_m5 = count_m15 * 3
    elif days:
        start_dt, end_dt = None, None
        cal_days = days
        count_m15 = days * 96
        count_m5 = days * 288
    else:
        client.shutdown()
        return {"error": "Must specify either --days or --start-date"}

    # Print header
    print("\n" + "=" * 80)
    print(f"  BROOKS BACKTEST: {symbol}")
    print("=" * 80)
    print(f"  Config Hash: {config.get_hash()}")
    print(f"  Regime Filter: {'ON' if config.regime_filter else 'OFF'}")
    if config.regime_filter:
        print(f"  Chop Threshold: {config.regime_params.chop_threshold}")
    print(f"  Max Trades/Day: {config.guardrails.max_trades_per_day}")
    print(f"  Costs: {config.costs_per_trade_r:.4f}R per trade")
    print("=" * 80)

    # Fetch data
    logger.info("Fetching data...")
    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data.empty or m5_data.empty:
        client.shutdown()
        return {"error": "Empty data"}

    m15_data = _normalize_ohlc(m15_data, name="M15")
    m5_data = _normalize_ohlc(m5_data, name="M5")

    # Filter to date range
    if start_dt and end_dt:
        m15_data = m15_data.loc[max(start_dt, m15_data.index[0]) : min(end_dt, m15_data.index[-1])]
        m5_data = m5_data.loc[max(start_dt, m5_data.index[0]) : min(end_dt, m5_data.index[-1])]

    period_start = m15_data.index[0] if len(m15_data) else None
    period_end = m15_data.index[-1] if len(m15_data) else None
    actual_cal_days = (period_end - period_start).days if period_start and period_end else 0

    # Regime & trend detection
    if config.regime_filter:
        regime_series = detect_regime_series(m15_data, config.regime_params)
        regime_data = pd.DataFrame({"regime": regime_series}, index=m15_data.index)

    trend_series = infer_trend_m15_series(m15_data, config.trend_params)
    m15_trends = pd.DataFrame({"trend": trend_series}, index=m15_data.index)

    # Merge to M5
    m5_ts = m5_data.reset_index().rename(columns={"index": "ts"})
    trend_df = m15_trends.reset_index().rename(columns={"index": "ts"})
    merged = pd.merge_asof(
        m5_ts.sort_values("ts"),
        trend_df.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    m5_data = m5_data.copy()
    m5_data["trend"] = merged["trend"].values

    if config.regime_filter:
        regime_df = regime_data.reset_index().rename(columns={"index": "ts"})
        merged_regime = pd.merge_asof(
            m5_ts.sort_values("ts"),
            regime_df.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        m5_data["regime"] = merged_regime["regime"].values

    # Trade planning
    planned_trades: list[PlannedTrade] = []
    total_bars = len(m5_data)
    segments = []
    current_trend, current_regime, segment_start = None, None, 50

    for i in range(50, total_bars):
        trend_val = m5_data.iloc[i]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None
        regime_val = (
            m5_data.iloc[i].get("regime", MarketRegime.UNKNOWN) if config.regime_filter else None
        )

        if side != current_trend or (config.regime_filter and regime_val != current_regime):
            if current_trend is not None and segment_start < i:
                segments.append((segment_start, i, current_trend, current_regime))
            current_trend, current_regime, segment_start = side, regime_val, i

    skipped_choppy = 0
    for start_idx, end_idx, trend_side, regime_val in segments:
        if config.regime_filter and regime_val == MarketRegime.CHOPPY:
            skipped_choppy += 1
            continue
        if trend_side is None:
            continue

        segment_data = m5_data.iloc[max(0, start_idx - 10) : end_idx]
        trades = plan_h2l2_trades(segment_data, trend_side, spec, config.h2l2_params)
        for t in trades:
            if t.execute_ts >= m5_data.index[start_idx]:
                planned_trades.append(t)

    logger.info(f"Planned trades: {len(planned_trades)}, Choppy segments skipped: {skipped_choppy}")

    # Guardrails & selection
    in_session, _ = apply_guardrails(
        planned_trades,
        Guardrails(**{**config.guardrails.__dict__, "max_trades_per_day": 10000}),
    )
    selected, _ = select_top_per_ny_day(
        in_session,
        max_trades_day=config.guardrails.max_trades_per_day,
        tick_size=float(spec.tick_size),
    )
    final_trades, _ = apply_guardrails(selected, config.guardrails)

    # Simulation
    results_r, exit_ts_list = [], []
    for t in final_trades:
        raw_r, exit_ts = _simulate_trade_outcome(m5_data, t)
        results_r.append(raw_r - config.costs_per_trade_r)
        exit_ts_list.append(pd.to_datetime(exit_ts))

    if not results_r:
        client.shutdown()
        return {
            "error": "No trades",
            "planned": len(planned_trades),
            "skipped_choppy": skipped_choppy,
        }

    # Metrics
    res = pd.Series(results_r, index=[t.execute_ts for t in final_trades]).sort_index()
    equity_curve = res.cumsum()
    drawdown = equity_curve - equity_curve.cummax()

    winrate = float((res > 0).sum() / len(res))
    profit_factor = (
        float(res[res > 0].sum() / abs(res[res < 0].sum())) if (res < 0).any() else float("inf")
    )

    trades_df = _build_trades_dataframe(final_trades, results_r, exit_ts_list, m5_data)
    daily_pnl_r = _daily_series_from_trades(trades_df, period_start, period_end)
    mgr = _compute_metrics(trades_df, daily_pnl_r, trading_days_per_year)

    # Dashboard
    run_id = f"bt_{symbol.replace('.', '')}_{actual_cal_days}d_{datetime.now().strftime('%H%M%S')}"

    if dashboard != "none":
        try:
            from backtest.visualiser_v2 import generate_dashboard_v2

            generate_dashboard_v2(
                results_r=res,
                equity_curve=equity_curve,
                drawdown=drawdown,
                daily_pnl_r=daily_pnl_r,
                trades_df=trades_df,
                config=config,
                symbol=symbol,
                days=actual_cal_days,
                run_id=run_id,
                period_start=period_start,
                period_end=period_end,
                stats={
                    "trades": len(res),
                    "net_r": float(equity_curve.iloc[-1]),
                    "winrate": winrate,
                    **mgr,
                },
                price_series=m5_data["close"],
            )
        except Exception as e:
            logger.warning(f"Dashboard generation failed: {e}")

    client.shutdown()

    # Print summary
    print("\n" + "=" * 80)
    print("  BACKTEST RESULTS")
    print("=" * 80)
    print(
        f"  Period: {period_start.date() if period_start else 'N/A'} to {period_end.date() if period_end else 'N/A'} ({actual_cal_days} days)"
    )
    print(f"  Trades: {len(res)}")
    print(f"  Net R: {float(equity_curve.iloc[-1]):+.2f}R")
    print(f"  Winrate: {winrate * 100:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Daily Sharpe: {mgr.get('daily_sharpe_r', 0):.3f}")
    print(f"  Max DD (daily): {mgr.get('max_dd_r_daily', 0):.2f}R")
    print("=" * 80)

    return {
        "days": actual_cal_days,
        "period_start": str(period_start.date()) if period_start else None,
        "period_end": str(period_end.date()) if period_end else None,
        "trades": int(len(res)),
        "net_r": float(equity_curve.iloc[-1]),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "avg_r": float(res.mean()),
        **mgr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Brooks Backtest Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to strategy YAML")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Number of days (legacy)")
    parser.add_argument("--max-trades-day", type=int, help="Override max trades per day")
    parser.add_argument("--dashboard", choices=["v2", "v1", "none"], default="v2")
    parser.add_argument("--initial-capital", type=float, default=10000.0)

    args = parser.parse_args()

    config = StrategyConfig.load(args.config)

    if args.max_trades_day:
        # Create new guardrails with override
        from execution.guardrails import Guardrails

        new_guardrails = Guardrails(
            session_tz=config.guardrails.session_tz,
            day_tz=config.guardrails.day_tz,
            session_start=config.guardrails.session_start,
            session_end=config.guardrails.session_end,
            max_trades_per_day=args.max_trades_day,
        )
        # Rebuild config (frozen dataclass workaround)
        config = StrategyConfig(
            symbol=config.symbol,
            regime_filter=config.regime_filter,
            regime_params=config.regime_params,
            trend_params=config.trend_params,
            h2l2_params=config.h2l2_params,
            guardrails=new_guardrails,
            risk_pct=config.risk_pct,
            costs_per_trade_r=config.costs_per_trade_r,
        )

    result = run_backtest_from_config(
        config,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        dashboard=args.dashboard,
    )

    if "error" in result:
        logger.error(f"Backtest failed: {result['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
