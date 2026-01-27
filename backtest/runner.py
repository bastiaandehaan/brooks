# backtest/runner.py
"""
Brooks Backtest Runner - CONFIG-FIRST VERSION with DATE-BASED MODE
Zero-drift guarantee: Uses StrategyConfig.load() as single source of truth.

Usage:
    # PREFERRED: Date-based mode
    python backtest/runner.py --config config/strategies/us500_sniper.yaml --start-date 2024-01-24 --end-date 2026-01-24

    # Legacy: Bar-count mode (less transparent)
    python backtest/runner.py --config config/strategies/us500_sniper.yaml --days 340

    # Override max trades per day
    python backtest/runner.py --config config/strategies/us500_sniper.yaml --start-date 2024-01-24 --end-date 2026-01-24 --max-trades-day 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# Suppress spam logging
import logging as _log

from execution.guardrails import Guardrails, apply_guardrails
from strategies.config import StrategyConfig
from strategies.context import Trend, TrendParams, infer_trend_m15_series
from strategies.h2l2 import H2L2Params, PlannedTrade, Side, plan_h2l2_trades
from strategies.regime import MarketRegime, RegimeParams, detect_regime_series
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates

_log.getLogger("execution.guardrails").setLevel(_log.WARNING)
_log.getLogger("execution.selection").setLevel(_log.WARNING)

from execution.selection import select_top_per_ny_day

from backtest.config_formatter import format_frozen_config_text
from backtest.visualiser import generate_performance_report

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


def precalculate_trends(m15_df: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    logger.info("-> Calculating trends (vectorized)...")
    t0 = time.perf_counter()
    trend_series = infer_trend_m15_series(m15_df, params)
    out = pd.DataFrame({"trend": trend_series}, index=m15_df.index)
    logger.info("  Trend calc: %.2fs", time.perf_counter() - t0)
    return out


def _trend_to_side(trend: Trend) -> Side | None:
    if trend == Trend.BULL:
        return Side.LONG
    if trend == Trend.BEAR:
        return Side.SHORT
    return None


def _simulate_trade_outcome(
    m5_data: pd.DataFrame, t: PlannedTrade
) -> tuple[float, pd.Timestamp]:
    """Simulate trade with worst-case both-hit policy. Returns (R-value BEFORE costs, exit_ts)."""
    future = m5_data.loc[t.execute_ts :]
    last_ts = (
        pd.to_datetime(future.index[-1])
        if len(future)
        else pd.to_datetime(t.execute_ts)
    )

    for ts, bar in future.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        ts = pd.to_datetime(ts)

        if t.side == Side.LONG:
            hit_sl = low <= t.stop
            hit_tp = high >= t.tp
            if hit_sl and hit_tp:
                return -1.0, ts
            if hit_sl:
                return -1.0, ts
            if hit_tp:
                return 2.0, ts
        else:
            hit_sl = high >= t.stop
            hit_tp = low <= t.tp
            if hit_sl and hit_tp:
                return -1.0, ts
            if hit_sl:
                return -1.0, ts
            if hit_tp:
                return 2.0, ts

    return 0.0, last_ts


def _apply_costs(result_r: float, costs_r: float) -> float:
    return result_r - costs_r


def _build_trades_dataframe(
    final_trades: List[PlannedTrade],
    results_r: List[float],
    exit_ts_list: List[pd.Timestamp],
    *,
    m5_data: pd.DataFrame,
    ny_tz: str,
) -> pd.DataFrame:
    rows = []
    for trade, result, exit_ts in zip(final_trades, results_r, exit_ts_list):
        entry_ts = pd.to_datetime(trade.execute_ts)
        exit_ts = pd.to_datetime(exit_ts)

        regime_val = None
        chop_ratio = None
        try:
            row = m5_data.loc[trade.execute_ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            regime_val = row.get("regime", None)
            chop_ratio = row.get("chop_ratio", None)
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
                "regime_at_entry": str(regime_val) if regime_val is not None else None,
                "chop_ratio_at_entry": float(chop_ratio)
                if chop_ratio is not None
                else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    if df["exit_time"].dt.tz is None:
        df["ny_day"] = (
            df["exit_time"].dt.tz_localize("UTC").dt.tz_convert(ny_tz).dt.date
        )
    else:
        df["ny_day"] = df["exit_time"].dt.tz_convert(ny_tz).dt.date

    return df


def _max_consecutive_losses(trade_pnl: List[float]) -> int:
    max_run = 0
    run = 0
    for r in trade_pnl:
        if r < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def _daily_series_from_trades(
    trades_df: pd.DataFrame,
    *,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
    ny_tz: str,
) -> pd.Series:
    """Daily PnL in R units indexed by NY calendar day (date objects)."""
    if trades_df.empty:
        return pd.Series(dtype="float64")

    daily = trades_df.groupby("ny_day")["net_r"].sum().astype("float64")

    if start_dt is not None and end_dt is not None:
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)

        if s.tzinfo is None:
            s = s.tz_localize("UTC").tz_convert(ny_tz)
        else:
            s = s.tz_convert(ny_tz)

        if e.tzinfo is None:
            e = e.tz_localize("UTC").tz_convert(ny_tz)
        else:
            e = e.tz_convert(ny_tz)

        full_days = pd.date_range(s.date(), e.date(), freq="D").date
        daily = daily.reindex(full_days, fill_value=0.0)

    return daily


def _compute_manager_metrics_r_based(
    trades_df: pd.DataFrame,
    *,
    daily_pnl_r: pd.Series,
    trading_days_per_year: int,
    initial_capital: float,
) -> Dict[str, Any]:
    """Compute daily risk/return metrics on daily PnL in R-units."""
    out: Dict[str, Any] = {}
    if daily_pnl_r.empty:
        return out

    dp = (
        pd.Series(daily_pnl_r, dtype="float64")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    mu = float(dp.mean())
    sig = float(dp.std(ddof=1)) if len(dp) > 1 else 0.0
    daily_sharpe_r = (mu / sig) * np.sqrt(trading_days_per_year) if sig > 0 else 0.0

    downside = dp[dp < 0]
    dsig = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    daily_sortino_r = (mu / dsig) * np.sqrt(trading_days_per_year) if dsig > 0 else 0.0

    eq_r = dp.cumsum()
    run_max = eq_r.cummax()
    dd_r = eq_r - run_max
    max_dd_r_daily = float(dd_r.min())
    max_dd_pct_initial = (
        (max_dd_r_daily / initial_capital) * 100.0 if initial_capital else 0.0
    )

    best_day_r = float(dp.max())
    worst_day_r = float(dp.min())

    var_95_r = float(np.nanpercentile(dp.values, 5)) if len(dp) else 0.0
    cvar_95_r = float(dp[dp <= var_95_r].mean()) if len(dp) else 0.0

    skew_r = float(dp.skew()) if len(dp) > 2 else 0.0
    kurtosis_r = float(dp.kurtosis()) if len(dp) > 3 else 0.0

    pct_pos_days = float((dp > 0).mean() * 100.0) if len(dp) else 0.0

    underwater = dd_r < 0
    max_underwater_days = 0
    run = 0
    for v in underwater.astype(int).values:
        if v == 1:
            run += 1
            max_underwater_days = max(max_underwater_days, run)
        else:
            run = 0

    total_calendar_days = int(len(dp))
    days_with_trades = int((dp != 0).sum())
    trades_total = int(len(trades_df))
    trades_per_active_day = (
        (trades_total / days_with_trades) if days_with_trades else 0.0
    )
    trades_per_calendar_day = (
        (trades_total / total_calendar_days) if total_calendar_days else 0.0
    )

    out.update(
        {
            "daily_sharpe_r": daily_sharpe_r,
            "daily_sortino_r": daily_sortino_r,
            "max_dd_r_daily": max_dd_r_daily,
            "max_dd_pct_initial": max_dd_pct_initial,
            "best_day_r": best_day_r,
            "worst_day_r": worst_day_r,
            "var_95_r": var_95_r,
            "cvar_95_r": cvar_95_r,
            "skew_r": skew_r,
            "kurtosis_r": kurtosis_r,
            "pct_pos_days": pct_pos_days,
            "max_underwater_days": int(max_underwater_days),
            "calendar_days": total_calendar_days,
            "days_with_trades": days_with_trades,
            "trades_per_active_day": trades_per_active_day,
            "trades_per_calendar_day": trades_per_calendar_day,
        }
    )

    return out


def run_backtest_from_config(
    config: StrategyConfig,
    args: argparse.Namespace,
    days: int = None,
    start_date: str = None,
    end_date: str = None,
    *,
    initial_capital: float = 10000.0,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Run backtest using StrategyConfig.
    """
    symbol = config.symbol

    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        return {"error": "MT5 init failed"}

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return {"error": "Symbol not found"}

    # ===== DETERMINE BACKTEST MODE =====
    if start_date:
        period_mode = "DATE-BASED"
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = (
            pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")
        )
        cal_days = (end_dt - start_dt).days
        count_m15 = max(cal_days * 90, 10000)
        count_m5 = count_m15 * 3
        days_display = cal_days
    elif days:
        period_mode = "BAR-COUNT (legacy)"
        start_dt = None
        end_dt = None
        count_m15 = days * 96
        count_m5 = days * 288
        days_display = days
    else:
        client.shutdown()
        return {"error": "Must specify either --days or --start-date"}

    # ===== PRINT HEADER =====
    print("\n" + "=" * 80)
    print(f"  BROOKS BACKTEST: {symbol}")
    print("=" * 80)
    print(f"  Mode: {period_mode}")
    if start_dt and end_dt:
        print(f"  Requested period: {start_dt.date()} to {end_dt.date()}")
    print(f"  Config Hash: {config.get_hash()}")
    if config.regime_filter:
        print(
            f"   REGIME FILTER: ENABLED (chop_threshold={config.regime_params.chop_threshold})"
        )
    else:
        print("  WARNING:  REGIME FILTER: DISABLED")
    if config.costs_per_trade_r > 0:
        print(f"  COSTS: COSTS: {config.costs_per_trade_r:.4f}R per trade")
    print("=" * 80)

    # ===== FETCH DATA =====
    logger.info("-> Fetching data...")
    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data.empty or m5_data.empty:
        client.shutdown()
        return {"error": "Empty data"}

    m15_data = _normalize_ohlc(m15_data, name="M15")
    m5_data = _normalize_ohlc(m5_data, name="M5")

    # ===== FILTER TO DATE RANGE =====
    if start_dt and end_dt:
        data_start = m15_data.index[0]
        data_end = m15_data.index[-1]
        effective_start = max(start_dt, data_start)
        effective_end = min(end_dt, data_end)
        m15_data = m15_data.loc[effective_start:effective_end]
        m5_data = m5_data.loc[effective_start:effective_end]

    actual_start = m15_data.index[0] if len(m15_data) else None
    actual_end = m15_data.index[-1] if len(m15_data) else None
    actual_cal_days = (
        (actual_end - actual_start).days if actual_start and actual_end else 0
    )
    period_start, period_end = actual_start, actual_end

    # ===== REGIME & TREND DETECTION =====
    regime_data = None
    if config.regime_filter:
        regime_series = detect_regime_series(m15_data, config.regime_params)
        regime_data = pd.DataFrame({"regime": regime_series}, index=m15_data.index)

    m15_trends = precalculate_trends(m15_data, config.trend_params)

    # Merge trends to M5
    trend_series = m15_trends.reset_index().rename(columns={"index": "ts"})
    m5_ts = m5_data.reset_index().rename(columns={"index": "ts"})
    merged = pd.merge_asof(
        m5_ts.sort_values("ts"),
        trend_series.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    m5_data = m5_data.copy()
    m5_data["trend"] = merged["trend"].values

    if config.regime_filter and regime_data is not None:
        regime_series_df = regime_data.reset_index().rename(columns={"index": "ts"})
        merged_regime = pd.merge_asof(
            m5_ts.sort_values("ts"),
            regime_series_df.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        m5_data["regime"] = merged_regime["regime"].values

    # ===== TRADE PLANNING =====
    planned_trades: list[PlannedTrade] = []
    total_bars = len(m5_data)
    segments = []
    current_trend = None
    current_regime = None
    segment_start = 50

    for i in range(50, total_bars):
        trend_val = m5_data.iloc[i]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None
        regime_val = (
            m5_data.iloc[i].get("regime", MarketRegime.UNKNOWN)
            if config.regime_filter
            else None
        )

        if side != current_trend or (
            config.regime_filter and regime_val != current_regime
        ):
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

    # ===== GUARDRAILS & EXECUTION =====
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

    # ===== SIMULATION =====
    results_r, exit_ts_list = [], []
    for t in final_trades:
        raw_r, exit_ts = _simulate_trade_outcome(m5_data, t)
        results_r.append(_apply_costs(raw_r, config.costs_per_trade_r))
        exit_ts_list.append(pd.to_datetime(exit_ts))

    if not results_r:
        client.shutdown()
        return {"error": "No trades"}

    # ===== METRICS =====
    res = pd.Series(results_r, index=[t.execute_ts for t in final_trades]).sort_index()
    equity_curve = res.cumsum()
    drawdown = equity_curve - equity_curve.cummax()
    max_dd_r = float(drawdown.min())
    winrate = float((res > 0).sum() / len(res))
    profit_factor = (
        float(res[res > 0].sum() / abs(res[res < 0].sum()))
        if (res < 0).any()
        else float("inf")
    )
    mean_r = float(res.mean())
    std_r = float(res.std()) if len(res) > 1 else 0.0
    sharpe_trade = float(mean_r / std_r) if std_r > 0 else 0.0

    # Brooks & side metrics
    long_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.LONG]
    short_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.SHORT]

    trades_df = _build_trades_dataframe(
        final_trades, results_r, exit_ts_list, m5_data=m5_data, ny_tz=NY_TZ
    )
    daily_pnl_r = _daily_series_from_trades(
        trades_df, start_dt=period_start, end_dt=period_end, ny_tz=NY_TZ
    )
    mgr = _compute_manager_metrics_r_based(
        trades_df,
        daily_pnl_r=daily_pnl_r,
        trading_days_per_year=trading_days_per_year,
        initial_capital=initial_capital,
    )

    # ===== OUTPUT & DASHBOARD =====
    symbol_clean = symbol.replace(".cash", "").replace(".", "")
    run_id = f"bt_{symbol_clean}_{actual_cal_days}d_{datetime.now().strftime('%H%M%S')}"

    stats = {
        "trades": len(res),
        "net_r": float(equity_curve.iloc[-1]),
        "winrate_pct": winrate * 100,
        "profit_factor": profit_factor,
        "max_dd_r_trade": max_dd_r,
        "avg_r": mean_r,
    }
    stats.update(mgr)

    # Dashboard logic
    dash_fn, _ = _get_dashboard_fn(args.dashboard)
    if dash_fn is not None:
        dash_fn(
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
            stats=stats,
            price_series=m5_data["close"],
        )

    # PATCH APPLIED START: Risk Analysis & Final Return
    if trades_df is not None and not trades_df.empty:
        from backtest.ftmo_risk_report import generate_ftmo_risk_report

        logger.info("\nRunning comprehensive risk analysis...")
        recommendation = generate_ftmo_risk_report(
            trades_df=trades_df,
            equity_curve_r=equity_curve,
            current_risk_pct=config.risk_pct,
            current_max_trades_day=config.guardrails.max_trades_per_day,
            n_monte_carlo_sims=10000,
        )

        if hasattr(args, "export_risk_report") and args.export_risk_report:
            _safe_export_risk_report(recommendation, args.export_risk_report)
            logger.info("Wrote risk report: %s", args.export_risk_report)

    client.shutdown()

    result = {
        "days": actual_cal_days,
        "period_start": str(period_start.date()) if period_start else None,
        "period_end": str(period_end.date()) if period_end else None,
        "trades": int(len(res)),
        "net_r": float(equity_curve.iloc[-1]),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "max_dd_r_trade": float(max_dd_r),
        "avg_r": float(mean_r),
        "trade_sharpe": float(sharpe_trade),
        **mgr,
    }

    if "recommendation" in locals():
        result.update(
            {
                "scaling_recommendation": recommendation,
                "can_scale_risk": recommendation.can_scale_risk,
                "max_safe_risk_pct": recommendation.max_safe_risk_pct,
                "can_increase_frequency": recommendation.can_increase_frequency,
                "max_safe_trades_per_day": recommendation.max_safe_trades_per_day,
            }
        )

    return result
    # PATCH APPLIED END


def _get_dashboard_fn(dashboard: str):
    if dashboard == "none":
        return None, "none"
    if dashboard == "v1":
        return generate_performance_report, "v1"
    try:
        from backtest.visualiser_v2 import generate_dashboard_v2

        return generate_dashboard_v2, "v2"
    except Exception as e:
        return generate_performance_report, "v1"


def _safe_export_risk_report(recommendation: Any, path: str) -> None:
    import json
    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        asdict(recommendation)
        if is_dataclass(recommendation)
        else getattr(recommendation, "__dict__", {"val": str(recommendation)})
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brooks Backtest Runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--days", type=int)
    parser.add_argument("--max-trades-day", type=int)
    parser.add_argument("--dashboard", choices=["v2", "v1", "none"], default="v2")
    parser.add_argument("--export-risk-report", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--trading-days-year", type=int, default=252)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = StrategyConfig.load(args.config)

    if args.max_trades_day:
        config.guardrails.max_trades_per_day = args.max_trades_day

    result = run_backtest_from_config(
        config,
        args=args,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        trading_days_per_year=args.trading_days_year,
    )

    if "error" in result:
        logger.error("Backtest failed: %s", result["error"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
