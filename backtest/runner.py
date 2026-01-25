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
    logger.info("→ Calculating trends (vectorized)...")
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


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> tuple[float, pd.Timestamp]:
    """Simulate trade with worst-case both-hit policy. Returns (R-value BEFORE costs, exit_ts)."""
    future = m5_data.loc[t.execute_ts :]
    last_ts = pd.to_datetime(future.index[-1]) if len(future) else pd.to_datetime(t.execute_ts)

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
                "chop_ratio_at_entry": float(chop_ratio) if chop_ratio is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    if df["exit_time"].dt.tz is None:
        df["ny_day"] = df["exit_time"].dt.tz_localize("UTC").dt.tz_convert(ny_tz).dt.date
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

    dp = pd.Series(daily_pnl_r, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
    max_dd_pct_initial = (max_dd_r_daily / initial_capital) * 100.0 if initial_capital else 0.0

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
    trades_per_active_day = (trades_total / days_with_trades) if days_with_trades else 0.0
    trades_per_calendar_day = (trades_total / total_calendar_days) if total_calendar_days else 0.0

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
    days: int = None,
    start_date: str = None,
    end_date: str = None,
    *,
    initial_capital: float = 10000.0,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Run backtest using StrategyConfig.

    Args:
        config: Strategy configuration
        days: Number of days (legacy bar-count mode)
        start_date: Explicit start date YYYY-MM-DD (preferred)
        end_date: Explicit end date YYYY-MM-DD (default: today)
        initial_capital: Starting account size
        trading_days_per_year: For Sharpe annualization

    Returns:
        Dictionary with backtest results
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
        # DATE-BASED MODE (preferred)
        period_mode = "DATE-BASED"
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")

        # Calculate bars needed
        # US500 trades ~23h/day, 5 days/week = ~90 bars/calendar day
        cal_days = (end_dt - start_dt).days
        count_m15 = max(cal_days * 90, 10000)  # Minimum 10k bars
        count_m5 = count_m15 * 3

        days_display = cal_days  # For display purposes

    elif days:
        # LEGACY BAR-COUNT MODE
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
        print(f"  📎 REGIME FILTER: ENABLED (chop_threshold={config.regime_params.chop_threshold})")
    else:
        print("  ⚠️  REGIME FILTER: DISABLED")
    if config.costs_per_trade_r > 0:
        print(f"  💸 COSTS: {config.costs_per_trade_r:.4f}R per trade")
    print("=" * 80)
    print("\n📊 STRATEGY PARAMETERS:")
    print(
        f"  Context    : min_slope={config.trend_params.min_slope:.2f}, ema_period={config.trend_params.ema_period}"
    )
    print(
        f"  H2/L2      : pullback={config.h2l2_params.pullback_bars}, close_frac={config.h2l2_params.signal_close_frac:.2f}"
    )
    print(
        f"  Risk       : stop_buffer={config.h2l2_params.stop_buffer:.1f}, min_risk={config.h2l2_params.min_risk_price_units:.1f}"
    )
    print(
        f"  Execution  : cooldown={config.h2l2_params.cooldown_bars}bars, max_trades_day={config.guardrails.max_trades_per_day}"
    )
    print()

    # ===== FETCH DATA =====
    logger.info("→ Fetching data...")
    logger.info(f"  Estimated bars: M15={count_m15:,d}, M5={count_m5:,d}")

    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data.empty or m5_data.empty:
        logger.warning("Empty data!")
        client.shutdown()
        return {"error": "Empty data"}

    m15_data = _normalize_ohlc(m15_data, name="M15")
    m5_data = _normalize_ohlc(m5_data, name="M5")

    # ===== FILTER TO DATE RANGE (if date-based) =====
    if start_dt and end_dt:
        # Check if requested period is within available data
        data_start = m15_data.index[0] if len(m15_data) > 0 else None
        data_end = m15_data.index[-1] if len(m15_data) > 0 else None

        if data_start and data_end:
            # Check if requested range overlaps with available data
            if end_dt < data_start or start_dt > data_end:
                logger.error(
                    "❌ Requested period (%s to %s) is OUTSIDE available data (%s to %s)",
                    start_dt.date(),
                    end_dt.date(),
                    data_start.date(),
                    data_end.date(),
                )
                logger.error(
                    "   Available data spans: %s to %s",
                    data_start.date(),
                    data_end.date(),
                )
                logger.error("   Suggestion: Use --days %d instead", (data_end - data_start).days)
                client.shutdown()
                return {
                    "error": f"No data overlap. Available: {data_start.date()} to {data_end.date()}"
                }

            # Adjust to actual overlap
            effective_start = max(start_dt, data_start)
            effective_end = min(end_dt, data_end)

            if effective_start != start_dt or effective_end != end_dt:
                logger.warning("⚠️  Adjusting period to available data:")
                logger.warning("   Requested: %s to %s", start_dt.date(), end_dt.date())
                logger.warning("   Available: %s to %s", data_start.date(), data_end.date())
                logger.warning(
                    "   Using:     %s to %s",
                    effective_start.date(),
                    effective_end.date(),
                )
                start_dt = effective_start
                end_dt = effective_end

            m15_data = m15_data.loc[start_dt:end_dt]
            m5_data = m5_data.loc[start_dt:end_dt]
        else:
            logger.error("❌ No data fetched")
            client.shutdown()
            return {"error": "No data available"}

    # ===== VERIFICATION: What did we actually get? =====
    actual_start = m15_data.index[0] if len(m15_data) else None
    actual_end = m15_data.index[-1] if len(m15_data) else None
    actual_cal_days = (actual_end - actual_start).days if actual_start and actual_end else 0
    actual_bars_m15 = len(m15_data)
    actual_bars_m5 = len(m5_data)

    print("=" * 80)
    print("📅 DATA PERIOD VERIFICATION")
    print("=" * 80)
    if start_dt and end_dt:
        print(f"  Requested: {start_dt.date()} to {end_dt.date()}")
    print(
        f"  Actual:    {actual_start.date() if actual_start else 'N/A'} to {actual_end.date() if actual_end else 'N/A'}"
    )
    print(f"  Calendar days: {actual_cal_days}")
    print(f"  M15 bars: {actual_bars_m15:,d}")
    print(f"  M5 bars: {actual_bars_m5:,d}")

    if actual_cal_days > 0:
        bars_per_day = actual_bars_m15 / actual_cal_days
        trading_days_est = actual_bars_m15 / 26  # ~26 bars per trading day (6.5h session)
        print(f"  Avg bars/day: {bars_per_day:.1f}")
        print(f"  Trading days (est): {trading_days_est:.0f}")

    # Warning if major mismatch
    if days and abs(actual_cal_days - days) > days * 0.15:  # >15% difference
        logger.warning(
            "⚠️  MISMATCH: Requested %d days but got %d calendar days",
            days,
            actual_cal_days,
        )
        logger.warning("⚠️  This is expected for 24/5 markets (bars/day varies with gaps)")

    print("=" * 80)
    print()

    # Store actual period for later use
    period_start = actual_start
    period_end = actual_end

    # ===== REGIME DETECTION =====
    regime_data = None
    if config.regime_filter:
        logger.info("→ Calculating market regime (vectorized)...")
        t0 = time.perf_counter()

        regime_series = detect_regime_series(m15_data, config.regime_params)

        high = m15_data["high"].astype(float)
        low = m15_data["low"].astype(float)
        close = m15_data["close"].astype(float)

        bar_range = high - low
        atr = bar_range.rolling(
            config.regime_params.atr_period, min_periods=config.regime_params.atr_period
        ).mean()
        avg_atr = atr.rolling(
            config.regime_params.range_period,
            min_periods=config.regime_params.range_period,
        ).mean()

        range_high = close.rolling(
            config.regime_params.range_period,
            min_periods=config.regime_params.range_period,
        ).max()
        range_low = close.rolling(
            config.regime_params.range_period,
            min_periods=config.regime_params.range_period,
        ).min()
        price_range = range_high - range_low

        threshold_range = config.regime_params.chop_threshold * avg_atr
        chop_ratio = (price_range / threshold_range).fillna(0.0)

        regime_data = pd.DataFrame(
            {"regime": regime_series, "chop_ratio": chop_ratio}, index=m15_data.index
        )

        logger.info("  Regime calc: %.2fs", time.perf_counter() - t0)

    # ===== TREND DETECTION =====
    m15_trends = precalculate_trends(m15_data, config.trend_params)

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
        m5_data["chop_ratio"] = merged_regime["chop_ratio"].values

    # ===== PRINT DISTRIBUTIONS =====
    trend_counts = m5_data["trend"].value_counts()
    bull_bars = int(trend_counts.get(Trend.BULL, 0))
    bear_bars = int(trend_counts.get(Trend.BEAR, 0))
    none_bars = int(m5_data["trend"].isna().sum())

    print("📈 TREND DISTRIBUTION:")
    print(f"  Bull  : {bull_bars:5d} bars ({bull_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Bear  : {bear_bars:5d} bars ({bear_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Range : {none_bars:5d} bars ({none_bars / len(m5_data) * 100:5.1f}%)")

    if config.regime_filter:
        regime_counts = m5_data["regime"].value_counts()
        trending_bars = int(regime_counts.get(MarketRegime.TRENDING, 0))
        choppy_bars = int(regime_counts.get(MarketRegime.CHOPPY, 0))
        unknown_bars = int(regime_counts.get(MarketRegime.UNKNOWN, 0))
        print("\n📎 REGIME DISTRIBUTION:")
        print(f"  Trending : {trending_bars:5d} bars ({trending_bars / len(m5_data) * 100:5.1f}%)")
        print(f"  Choppy   : {choppy_bars:5d} bars ({choppy_bars / len(m5_data) * 100:5.1f}%)")
        print(f"  Unknown  : {unknown_bars:5d} bars ({unknown_bars / len(m5_data) * 100:5.1f}%)")
    print()

    # ===== TRADE PLANNING =====
    planned_trades: list[PlannedTrade] = []

    logger.info("→ Planning trades (segment-based)...")
    t_plan0 = time.perf_counter()

    total_bars = len(m5_data)

    segments = []
    current_trend = None
    current_regime = None
    segment_start = 50

    for i in range(50, total_bars):
        trend_val = m5_data.iloc[i]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None

        regime_val = None
        if config.regime_filter:
            regime_val = m5_data.iloc[i].get("regime", MarketRegime.UNKNOWN)

        should_break = side != current_trend
        if config.regime_filter:
            should_break = should_break or (regime_val != current_regime)

        if should_break:
            if current_trend is not None and segment_start < i:
                segments.append((segment_start, i, current_trend, current_regime))
            current_trend = side
            current_regime = regime_val
            segment_start = i

    if current_trend is not None and segment_start < total_bars:
        segments.append((segment_start, total_bars, current_trend, current_regime))

    skipped_choppy = 0
    processed_segments = 0

    for start_idx, end_idx, trend_side, regime_val in segments:
        if config.regime_filter and regime_val == MarketRegime.CHOPPY:
            skipped_choppy += 1
            continue
        if trend_side is None:
            continue

        processed_segments += 1
        lookback_start = max(0, start_idx - 10)
        segment_data = m5_data.iloc[lookback_start:end_idx]

        trades = plan_h2l2_trades(segment_data, trend_side, spec, config.h2l2_params)

        segment_start_ts = m5_data.index[start_idx]
        for t in trades:
            if t.execute_ts >= segment_start_ts:
                planned_trades.append(t)

    planning_time = time.perf_counter() - t_plan0

    if config.regime_filter:
        logger.info(
            "  Planning: %.2fs, %d segments (%d choppy skipped, %d processed) → %d candidates",
            planning_time,
            len(segments),
            skipped_choppy,
            processed_segments,
            len(planned_trades),
        )
    else:
        logger.info(
            "  Planning: %.2fs, %d segments → %d candidates",
            planning_time,
            len(segments),
            len(planned_trades),
        )

    # ===== GUARDRAILS + SELECTION =====
    logger.info("→ Applying guardrails...")
    g_session = Guardrails(
        session_tz=config.guardrails.session_tz,
        day_tz=config.guardrails.day_tz,
        session_start=config.guardrails.session_start,
        session_end=config.guardrails.session_end,
        max_trades_per_day=10_000,  # No daily limit for session filter
    )
    in_session, rejected1 = apply_guardrails(planned_trades, g_session)

    selected, sel_stats = select_top_per_ny_day(
        in_session,
        max_trades_day=config.guardrails.max_trades_per_day,
        tick_size=float(spec.tick_size),
    )

    final_trades, rejected2 = apply_guardrails(selected, config.guardrails)
    final_trades = sorted(final_trades, key=lambda t: pd.to_datetime(t.execute_ts))

    print("📎 TRADE PIPELINE:")
    print(f"  Candidates    : {len(planned_trades):4d}")
    if config.regime_filter:
        print(f"  Choppy skipped: {skipped_choppy:4d} segments")
    print(f"  In session    : {len(in_session):4d}")
    print(f"  After select  : {len(selected):4d}")
    print(f"  Final         : {len(final_trades):4d}")
    print()

    # ===== SIMULATION =====
    results_r: list[float] = []
    exit_ts_list: list[pd.Timestamp] = []

    for t in final_trades:
        raw_r, exit_ts = _simulate_trade_outcome(m5_data, t)
        net_r = _apply_costs(raw_r, config.costs_per_trade_r)
        results_r.append(net_r)
        exit_ts_list.append(pd.to_datetime(exit_ts))

    if not results_r:
        logger.info("⚠️  No trades executed.")
        client.shutdown()
        return {"error": "No trades"}

    # ===== METRICS CALCULATION =====
    trade_ts = pd.to_datetime([t.execute_ts for t in final_trades])
    res = pd.Series(results_r, index=trade_ts, dtype="float64").sort_index()

    equity_curve = res.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd_r = float(drawdown.min())

    dd_bars = 0
    max_dd_bars = 0
    for dd_val in drawdown.values:
        if dd_val < 0:
            dd_bars += 1
            max_dd_bars = max(max_dd_bars, dd_bars)
        else:
            dd_bars = 0

    wins = float(res[res > 0].sum())
    losses = float(-res[res < 0].sum())
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    mean_r = float(res.mean())
    std_r = float(res.std(ddof=1)) if len(res) > 1 else 0.0
    sharpe_trade = float(mean_r / std_r) if std_r > 0 else 0.0

    recovery_factor = float(equity_curve.iloc[-1] / abs(max_dd_r)) if max_dd_r < 0 else float("inf")
    annual_r = (
        float(equity_curve.iloc[-1]) * (252.0 / float(actual_cal_days))
        if actual_cal_days > 0
        else 0.0
    )
    mar_ratio = annual_r / abs(max_dd_r) if max_dd_r < 0 else float("inf")

    winrate = float((res > 0).sum() / len(res)) if len(res) else 0.0
    avg_win_r = float(res[res > 0].mean()) if (res > 0).any() else 0.0
    avg_loss_r = float(res[res < 0].mean()) if (res < 0).any() else 0.0
    payoff_ratio = (avg_win_r / abs(avg_loss_r)) if avg_loss_r < 0 else float("inf")
    max_consec_losses = _max_consecutive_losses(list(res.values))

    long_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.LONG]
    short_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.SHORT]
    long_wr = (sum(1 for r in long_results if r > 0) / len(long_results)) if long_results else 0.0
    short_wr = (
        (sum(1 for r in short_results if r > 0) / len(short_results)) if short_results else 0.0
    )

    trades_df = _build_trades_dataframe(
        final_trades,
        results_r,
        exit_ts_list,
        m5_data=m5_data,
        ny_tz=NY_TZ,
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

    # ===== PRINT RESULTS =====
    print("=" * 80)
    print("  📊 RESULTS")
    print("=" * 80)

    if config.costs_per_trade_r > 0:
        total_cost = config.costs_per_trade_r * len(res)
        gross_profit = float(equity_curve.iloc[-1]) + total_cost
        print("\n💸 COSTS IMPACT:")
        print(f"  Cost per trade : {config.costs_per_trade_r:.4f}R")
        print(f"  Total cost     : {total_cost:.2f}R over {len(res)} trades")
        print(f"  Gross profit   : {gross_profit:+.2f}R (before costs)")
        print(f"  Net profit     : {float(equity_curve.iloc[-1]):+.2f}R (after costs)")
        if gross_profit != 0:
            print(
                f"  Impact         : {-total_cost:.2f}R ({-total_cost / gross_profit * 100:.1f}% reduction)"
            )

    print("\n💰 PERFORMANCE:")
    print(f"  Trades        : {len(res):4d}")
    print(f"  Net R         : {float(equity_curve.iloc[-1]):+7.2f}R")
    print(f"  Avg R/trade   : {mean_r:+7.4f}R")
    print(f"  Trade Sharpe  : {sharpe_trade:7.3f}  (trade-level, legacy)")
    print(f"  Profit Factor : {profit_factor:7.2f}")

    print("\n🎯 BROOKS METRICS:")
    print(f"  Expectancy     : {mean_r:+7.4f}R per trade")
    print(f"  Recovery Factor: {recovery_factor:7.2f}")
    print(f"  MAR Ratio      : {mar_ratio:7.2f}")
    print(f"  Annual R est.  : {annual_r:+7.2f}R ({actual_cal_days} days → 252 days)")

    if mgr:
        print("\n📌 MANAGER METRICS (R/day):")
        print(f"  Daily Sharpe (R/day)   : {mgr.get('daily_sharpe_r', 0.0):7.3f}")
        print(f"  Daily Sortino (R/day)  : {mgr.get('daily_sortino_r', 0.0):7.3f}")
        print(f"  Max DD (daily, R)      : {mgr.get('max_dd_r_daily', 0.0):7.2f}R")
        print(f"  Max DD (% of initial)  : {mgr.get('max_dd_pct_initial', 0.0):7.3f}%")

    print("\n📈 WIN/LOSS:")
    print(f"  Winrate       : {winrate * 100:6.2f}%")
    print(f"  Winners       : {int((res > 0).sum()):4d}")
    print(f"  Losers        : {int((res < 0).sum()):4d}")
    print(f"  Avg Win       : {avg_win_r:+7.4f}R")
    print(f"  Avg Loss      : {avg_loss_r:+7.4f}R")
    print(f"  Payoff Ratio  : {payoff_ratio:7.2f}")
    print(f"  Max consec losses: {max_consec_losses:4d}")

    print("\n⚖️ SIDE BREAKDOWN:")
    print(f"  Long trades   : {len(long_results):4d} (WR: {long_wr * 100:5.1f}%)")
    print(f"  Short trades  : {len(short_results):4d} (WR: {short_wr * 100:5.1f}%)")
    print("\n" + "=" * 80 + "\n")

    # ===== GENERATE DASHBOARD =====
    price_series = m15_data["close"] if "close" in m15_data.columns else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build descriptive filename
    symbol_clean = symbol.replace(".cash", "").replace(".", "")
    start_str = period_start.strftime("%Y-%m-%d") if period_start else "unknown"
    end_str = period_end.strftime("%Y-%m-%d") if period_end else "unknown"

    net_r_str = f"{float(equity_curve.iloc[-1]):+.0f}R"
    wr_str = f"WR{winrate * 100:.0f}"
    tpd = config.guardrails.max_trades_per_day

    run_id = f"bt_{symbol_clean}_{actual_cal_days}d_{start_str}_to_{end_str}_{tpd}tpd_{net_r_str}_{wr_str}_{timestamp}"

    stats: Dict[str, Any] = {
        "trades": int(len(res)),
        "net_r": float(equity_curve.iloc[-1]),
        "avg_r": float(mean_r),
        "winrate_pct": float(winrate * 100.0),
        "profit_factor": float(profit_factor),
        "max_dd_r_trade": float(max_dd_r),
        "trade_sharpe": float(sharpe_trade),
        "recovery_factor": float(recovery_factor),
        "mar_ratio": float(mar_ratio),
        "annual_r": float(annual_r),
        "costs_per_trade_r": float(config.costs_per_trade_r),
        "total_cost_r": float(config.costs_per_trade_r * len(res)),
        "regime_filter": bool(config.regime_filter),
        "choppy_segments_skipped": int(skipped_choppy if config.regime_filter else 0),
        "avg_win_r": float(avg_win_r),
        "avg_loss_r": float(avg_loss_r),
        "payoff_ratio": float(payoff_ratio),
        "max_consec_losses": int(max_consec_losses),
        "long_trades": len(long_results),
        "short_trades": len(short_results),
    }

    stats.update(mgr)

    def _pf_and_net(values: List[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        s = pd.Series(values, dtype="float64")
        w = float(s[s > 0].sum())
        l = float(-s[s < 0].sum())
        pf = (w / l) if l > 0 else float("inf")
        return pf, float(s.sum())

    pf_long, net_long = _pf_and_net(long_results)
    pf_short, net_short = _pf_and_net(short_results)
    stats["pf_long"] = pf_long
    stats["net_r_long"] = net_long
    stats["pf_short"] = pf_short
    stats["net_r_short"] = net_short

    generate_performance_report(
        results_r=res,
        equity_curve=equity_curve,
        drawdown=drawdown,
        symbol=symbol,
        days=actual_cal_days,
        run_id=run_id,
        command=f"Config: {config.get_hash()} | Period: {start_str} to {end_str}",
        period_start=period_start,
        period_end=period_end,
        stats=stats,
        price_series=price_series,
        daily_pnl=daily_pnl_r,
    )

    # After generating standard report, add risk analysis
    if trades_df is not None and not trades_df.empty:
        from backtest.ftmo_risk_report import generate_ftmo_risk_report

        logger.info("\n🔬 Running comprehensive risk analysis...")

        recommendation = generate_ftmo_risk_report(
            trades_df=trades_df,
            equity_curve_r=equity_curve,
            current_risk_pct=config.risk_pct,
            current_max_trades_day=config.guardrails.max_trades_per_day,
            n_monte_carlo_sims=10000,
        )

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

    # Add to return dict
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


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with date-based mode as primary"""
    parser = argparse.ArgumentParser(
        description="Brooks Backtest Runner - Date-Based Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

RECOMMENDED: Date-based mode (explicit period)
python backtest/runner.py --config config/strategies/us500_sniper.yaml --start-date 2024-01-24 --end-date 2026-01-24

Test with 2 trades/day:
python backtest/runner.py --config config/strategies/us500_sniper.yaml --start-date 2024-01-24 --end-date 2026-01-24 --max-trades-day 2

Legacy: Bar-count mode (less transparent)
python backtest/runner.py --config config/strategies/us500_sniper.yaml --days 340
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config YAML/JSON (REQUIRED)",
    )

    # Date-based mode (PREFERRED)
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date YYYY-MM-DD (e.g., 2024-01-24)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date YYYY-MM-DD (default: today)",
    )

    # Legacy bar-count mode
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days (legacy mode, less accurate)",
    )

    # Overrides
    parser.add_argument(
        "--max-trades-day",
        type=int,
        help="Override max trades per day from config",
    )

    # Other parameters
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--trading-days-year", type=int, default=252)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    logger.info("Loading config from: %s", args.config)
    config = StrategyConfig.load(args.config)

    # Apply CLI overrides
    if args.max_trades_day:
        logger.warning("⚠️  OVERRIDE: max_trades_per_day = %d", args.max_trades_day)
        config = StrategyConfig(
            symbol=config.symbol,
            regime_filter=config.regime_filter,
            regime_params=config.regime_params,
            trend_params=config.trend_params,
            h2l2_params=config.h2l2_params,
            guardrails=Guardrails(
                session_tz=config.guardrails.session_tz,
                day_tz=config.guardrails.day_tz,
                session_start=config.guardrails.session_start,
                session_end=config.guardrails.session_end,
                max_trades_per_day=args.max_trades_day,
            ),
            risk_pct=config.risk_pct,
            costs_per_trade_r=config.costs_per_trade_r,
        )

    # Run backtest
    result = run_backtest_from_config(
        config,
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
