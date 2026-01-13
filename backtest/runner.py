# backtest/runner.py
"""
Brooks Backtest Runner - WITH REGIME FILTER & COSTS & DAILY SHARPE
Skip choppy days, apply realistic trading costs, calculate proper Sharpe ratio
"""
import sys
import os
import time
import numpy as np
import argparse
import logging
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, Any, List

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import TrendParams, Trend, infer_trend_m15_series
from strategies.h2l2 import plan_h2l2_trades, H2L2Params, Side, PlannedTrade
from strategies.regime import RegimeParams, detect_regime_series, MarketRegime
from execution.guardrails import Guardrails, apply_guardrails
from utils.daily_sharpe_calculator import calculate_daily_sharpe

# Suppress guardrail spam logging
import logging as _log

_log.getLogger("execution.guardrails").setLevel(_log.WARNING)
from execution.selection import select_top_per_ny_day

# Suppress selection spam logging
_log.getLogger("execution.selection").setLevel(_log.WARNING)
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


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> float:
    """
    Simulate trade outcome with worst-case both-hit policy.
    Returns R-value BEFORE costs (costs applied later).
    """
    future = m5_data.loc[t.execute_ts:]
    for _, bar in future.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        if t.side == Side.LONG:
            hit_sl = low <= t.stop
            hit_tp = high >= t.tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0
            if hit_tp:
                return 2.0
        else:
            hit_sl = high >= t.stop
            hit_tp = low <= t.tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0
            if hit_tp:
                return 2.0
    return 0.0


def _apply_costs(result_r: float, costs_r: float) -> float:
    """
    Apply trading costs to a trade result.

    Costs are deducted from BOTH wins and losses:
    - Winner: +2R → +2R - costs
    - Loser: -1R → -1R - costs (makes it worse!)
    - Breakeven: 0R → -costs (turns into loss!)

    Args:
        result_r: Raw trade result in R
        costs_r: Cost per trade in R (e.g., 0.04R)

    Returns:
        Net result after costs
    """
    return result_r - costs_r


def _build_trades_dataframe(final_trades: List[PlannedTrade], results_r: List[float]) -> pd.DataFrame:
    """Build DataFrame of trade history for analysis."""
    trades_data = []
    for trade, result in zip(final_trades, results_r):
        trades_data.append({
            'entry_time': trade.execute_ts,
            'exit_time': trade.execute_ts + pd.Timedelta(hours=2),  # Estimate exit time
            'side': trade.side.value,
            'entry': trade.entry,
            'stop': trade.stop,
            'tp': trade.tp,
            'net_r': result,
            'reason': trade.reason
        })
    return pd.DataFrame(trades_data)


def run_backtest(
        symbol: str,
        days: int,
        max_trades_day: int = 2,
        # STRATEGY PARAMS
        min_slope: float = 0.15,
        ema_period: int = 20,
        pullback_bars: int = 3,
        signal_close_frac: float = 0.30,
        stop_buffer: float = 2.0,
        min_risk_price_units: float = 2.0,
        cooldown_bars: int = 10,
        # REGIME FILTER
        regime_filter: bool = False,
        chop_threshold: float = 2.5,
        # COSTS
        costs_per_trade_r: float = 0.0,
) -> Dict[str, Any]:
    """
    Backtest with optional regime filter and trading costs
    """
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        return {"error": "MT5 init failed"}

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return {"error": "Symbol not found"}

    print("\n" + "=" * 80)
    print(f"  BROOKS BACKTEST: {symbol} ({days} days)")
    if regime_filter:
        print(f"  🔎 REGIME FILTER: ENABLED (chop_threshold={chop_threshold})")
    else:
        print(f"  ⚠️  REGIME FILTER: DISABLED")
    if costs_per_trade_r > 0:
        print(f"  💸 COSTS: {costs_per_trade_r:.4f}R per trade")
    print("=" * 80)
    print(f"\n📊 STRATEGY PARAMETERS:")
    print(f"  Context    : min_slope={min_slope:.2f}, ema_period={ema_period}")
    print(f"  H2/L2      : pullback={pullback_bars}, close_frac={signal_close_frac:.2f}")
    print(f"  Risk       : stop_buffer={stop_buffer:.1f}, min_risk={min_risk_price_units:.1f}")
    print(f"  Execution  : cooldown={cooldown_bars}bars, max_trades_day={max_trades_day}")
    print()

    count_m5 = days * 288
    count_m15 = days * 96 * 2

    logger.info("→ Fetching data...")
    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data.empty or m5_data.empty:
        logger.warning("Empty data!")
        client.shutdown()
        return {"error": "Empty data"}

    m15_data = _normalize_ohlc(m15_data, name="M15")
    m5_data = _normalize_ohlc(m5_data, name="M5")

    logger.info("  M15: %d bars, M5: %d bars", len(m15_data), len(m5_data))

    # Calculate Regime (if enabled)
    regime_data = None
    if regime_filter:
        logger.info("→ Calculating market regime (vectorized)...")
        t0 = time.perf_counter()

        regime_params = RegimeParams(chop_threshold=chop_threshold)
        regime_series = detect_regime_series(m15_data, regime_params)

        # Calculate chop_ratio for stats
        high = m15_data["high"].astype(float)
        low = m15_data["low"].astype(float)
        close = m15_data["close"].astype(float)

        bar_range = high - low
        atr = bar_range.rolling(regime_params.atr_period, min_periods=regime_params.atr_period).mean()
        avg_atr = atr.rolling(regime_params.range_period, min_periods=regime_params.range_period).mean()

        range_high = close.rolling(regime_params.range_period, min_periods=regime_params.range_period).max()
        range_low = close.rolling(regime_params.range_period, min_periods=regime_params.range_period).min()
        price_range = range_high - range_low

        threshold_range = regime_params.chop_threshold * avg_atr
        chop_ratio = price_range / threshold_range
        chop_ratio = chop_ratio.fillna(0.0)

        regime_data = pd.DataFrame({
            "regime": regime_series,
            "chop_ratio": chop_ratio
        }, index=m15_data.index)

        logger.info("  Regime calc: %.2fs", time.perf_counter() - t0)

    # Trends
    m15_trends = precalculate_trends(
        m15_data,
        TrendParams(min_slope=min_slope, ema_period=ema_period)
    )

    # Merge trends to M5
    trend_series = m15_trends.copy()
    trend_series = trend_series.reset_index().rename(columns={"index": "ts"})
    m5_ts = m5_data.reset_index().rename(columns={"index": "ts"})
    merged = pd.merge_asof(
        m5_ts.sort_values("ts"),
        trend_series.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    m5_data = m5_data.copy()
    m5_data["trend"] = merged["trend"].values

    # Merge regime to M5 (if enabled)
    if regime_filter and regime_data is not None:
        regime_series = regime_data.reset_index().rename(columns={"index": "ts"})
        merged_regime = pd.merge_asof(
            m5_ts.sort_values("ts"),
            regime_series.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        m5_data["regime"] = merged_regime["regime"].values
        m5_data["chop_ratio"] = merged_regime["chop_ratio"].values

    # Trend distribution
    trend_counts = m5_data["trend"].value_counts()
    bull_bars = trend_counts.get(Trend.BULL, 0)
    bear_bars = trend_counts.get(Trend.BEAR, 0)
    none_bars = m5_data["trend"].isna().sum()

    print(f"📈 TREND DISTRIBUTION:")
    print(f"  Bull  : {bull_bars:5d} bars ({bull_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Bear  : {bear_bars:5d} bars ({bear_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Range : {none_bars:5d} bars ({none_bars / len(m5_data) * 100:5.1f}%)")

    # Regime distribution (if enabled)
    if regime_filter:
        regime_counts = m5_data["regime"].value_counts()
        trending_bars = regime_counts.get(MarketRegime.TRENDING, 0)
        choppy_bars = regime_counts.get(MarketRegime.CHOPPY, 0)
        unknown_bars = regime_counts.get(MarketRegime.UNKNOWN, 0)

        print(f"\n🔎 REGIME DISTRIBUTION:")
        print(f"  Trending : {trending_bars:5d} bars ({trending_bars / len(m5_data) * 100:5.1f}%)")
        print(f"  Choppy   : {choppy_bars:5d} bars ({choppy_bars / len(m5_data) * 100:5.1f}%)")
        print(f"  Unknown  : {unknown_bars:5d} bars ({unknown_bars / len(m5_data) * 100:5.1f}%)")

    print()

    # Strategy params
    strat_params = H2L2Params(
        pullback_bars=pullback_bars,
        signal_close_frac=signal_close_frac,
        min_risk_price_units=min_risk_price_units,
        stop_buffer=stop_buffer,
        cooldown_bars=cooldown_bars,
    )

    planned_trades: list[PlannedTrade] = []

    logger.info("→ Planning trades (segment-based with regime filter)...")
    t_plan0 = time.perf_counter()

    total_bars = len(m5_data)

    # Segment-based (with regime awareness)
    segments = []
    current_trend = None
    current_regime = None
    segment_start = 50

    for i in range(50, total_bars):
        trend_val = m5_data.iloc[i]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None

        # Check regime
        regime_val = None
        if regime_filter:
            regime_val = m5_data.iloc[i].get("regime", MarketRegime.UNKNOWN)

        # Segment breaks on trend or regime change
        should_break = (side != current_trend)
        if regime_filter:
            should_break = should_break or (regime_val != current_regime)

        if should_break:
            if current_trend is not None and segment_start < i:
                segments.append((segment_start, i, current_trend, current_regime))
            current_trend = side
            current_regime = regime_val
            segment_start = i

    if current_trend is not None and segment_start < total_bars:
        segments.append((segment_start, total_bars, current_trend, current_regime))

    # Track skipped segments
    skipped_choppy = 0
    processed_segments = 0

    for seg_idx, (start_idx, end_idx, trend_side, regime_val) in enumerate(segments):
        # Skip if regime is CHOPPY
        if regime_filter and regime_val == MarketRegime.CHOPPY:
            skipped_choppy += 1
            continue

        if trend_side is None:
            continue

        processed_segments += 1

        lookback_start = max(0, start_idx - 10)
        segment_data = m5_data.iloc[lookback_start:end_idx]

        trades = plan_h2l2_trades(segment_data, trend_side, spec, strat_params)

        segment_start_ts = m5_data.index[start_idx]
        for t in trades:
            if t.execute_ts >= segment_start_ts:
                planned_trades.append(t)

    planning_time = time.perf_counter() - t_plan0

    if regime_filter:
        logger.info(
            "  Planning: %.2fs, %d segments (%d choppy skipped, %d processed) → %d candidates",
            planning_time, len(segments), skipped_choppy, processed_segments, len(planned_trades)
        )
    else:
        logger.info(
            "  Planning: %.2fs, %d segments → %d candidates",
            planning_time, len(segments), len(planned_trades)
        )

    # Guardrails (silent mode)
    logger.info("→ Applying guardrails (session + daily limit)...")
    g_session = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=10_000,
    )
    in_session, rejected1 = apply_guardrails(planned_trades, g_session)

    selected, sel_stats = select_top_per_ny_day(
        in_session,
        max_trades_day=max_trades_day,
        tick_size=float(spec.tick_size),
    )

    g_all = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=max_trades_day,
    )
    final_trades, rejected2 = apply_guardrails(selected, g_all)

    print(f"🔎 TRADE PIPELINE:")
    print(f"  Candidates    : {len(planned_trades):4d}")
    if regime_filter:
        print(f"  Choppy skipped: {skipped_choppy:4d} segments")
    print(f"  In session    : {len(in_session):4d} (rejected: {len(rejected1)})")
    print(f"  After select  : {len(selected):4d} (days: {len(sel_stats)})")
    print(f"  Final         : {len(final_trades):4d}")
    print()

    # Simulate with costs
    results_r: list[float] = []
    for t in final_trades:
        raw_result = _simulate_trade_outcome(m5_data, t)
        net_result = _apply_costs(raw_result, costs_per_trade_r)
        results_r.append(net_result)

    if not results_r:
        logger.info("⚠️  No trades executed.")
        client.shutdown()
        return {"error": "No trades"}

    res = pd.Series(results_r, dtype="float64")
    equity_curve = res.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = float(drawdown.min())

    # DD duration
    dd_bars = 0
    max_dd_bars = 0
    for dd_val in drawdown:
        if dd_val < 0:
            dd_bars += 1
            max_dd_bars = max(max_dd_bars, dd_bars)
        else:
            dd_bars = 0

    wins = res[res > 0].sum()
    losses = -res[res < 0].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    mean_r = float(res.mean())
    std_r = float(res.std(ddof=1)) if len(res) > 1 else 0.0
    sharpe = float(mean_r / std_r) if std_r > 0 else 0.0

    # Brooks metrics
    recovery_factor = float(equity_curve.iloc[-1] / abs(max_dd)) if max_dd < 0 else float("inf")

    trading_days_in_sample = days
    annual_r = float(equity_curve.iloc[-1]) * (252.0 / trading_days_in_sample)
    mar_ratio = annual_r / abs(max_dd) if max_dd < 0 else float("inf")

    winrate = float((res > 0).sum() / len(res)) if len(res) else 0.0

    # Side breakdown
    long_trades = [t for t in final_trades if t.side == Side.LONG]
    short_trades = [t for t in final_trades if t.side == Side.SHORT]

    long_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.LONG]
    short_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.SHORT]

    long_wr = sum(1 for r in long_results if r > 0) / len(long_results) if long_results else 0
    short_wr = sum(1 for r in short_results if r > 0) / len(short_results) if short_results else 0

    # Build trades DataFrame for Daily Sharpe calculation
    trades_df = _build_trades_dataframe(final_trades, results_r)

    # Calculate Daily Sharpe
    daily_sharpe_metrics = calculate_daily_sharpe(
        trades_df[['exit_time', 'net_r']],
        initial_capital=10000.0,
        trading_days_per_year=252
    )

    # Output
    print("=" * 80)
    print("  📊 RESULTS")
    print("=" * 80)

    # Show costs impact if applied
    if costs_per_trade_r > 0:
        total_cost = costs_per_trade_r * len(res)
        gross_profit = float(equity_curve.iloc[-1]) + total_cost
        print(f"\n💸 COSTS IMPACT:")
        print(f"  Cost per trade : {costs_per_trade_r:.4f}R")
        print(f"  Total cost     : {total_cost:.2f}R over {len(res)} trades")
        print(f"  Gross profit   : {gross_profit:+.2f}R (before costs)")
        print(f"  Net profit     : {float(equity_curve.iloc[-1]):+.2f}R (after costs)")
        print(f"  Impact         : {-total_cost:.2f}R ({-total_cost / gross_profit * 100:.1f}% reduction)")

    print(f"\n💰 PERFORMANCE:")
    print(f"  Trades        : {len(res):4d}")
    print(f"  Net R         : {float(equity_curve.iloc[-1]):+7.2f}R")
    print(f"  Avg R/trade   : {mean_r:+7.4f}R")
    print(f"  Trade Sharpe  : {sharpe:7.3f}  (legacy, use Daily Sharpe)")
    print(f"  Daily Sharpe  : {daily_sharpe_metrics['daily_sharpe']:7.3f}  ⭐ (correct metric)")
    print(f"  Profit Factor : {profit_factor:7.2f}")

    print(f"\n🎯 BROOKS METRICS:")
    print(f"  Expectancy    : {mean_r:+7.4f}R per trade")
    print(f"  Recovery Factor: {recovery_factor:7.2f} (Net/MaxDD)")
    print(f"  MAR Ratio     : {mar_ratio:7.2f} (Annual/MaxDD)")
    print(f"  Annual R est. : {annual_r:+7.2f}R ({days} days → 252 days)")

    print(f"\n📉 DAILY RETURNS ANALYSIS:")
    print(f"  Daily Sharpe     : {daily_sharpe_metrics['daily_sharpe']:7.3f}")
    print(f"  Annualized Return: {daily_sharpe_metrics['annualized_return']:7.2f}%")
    print(f"  Annualized Vol   : {daily_sharpe_metrics['annualized_vol']:7.2f}%")
    print(f"  Trading Days     : {daily_sharpe_metrics['total_trading_days']:4d} calendar days")
    print(f"  Days w/ Trades   : {daily_sharpe_metrics['days_with_trades']:4d} active days")

    print(f"\n📈 WIN/LOSS:")
    print(f"  Winrate       : {winrate * 100:6.2f}%")
    print(f"  Winners       : {int((res > 0).sum()):4d} ({int((res > 0).sum()) / len(res) * 100:5.1f}%)")
    print(f"  Losers        : {int((res < 0).sum()):4d} ({int((res < 0).sum()) / len(res) * 100:5.1f}%)")
    print(f"  Breakeven     : {int((res == 0).sum()):4d}")

    print(f"\n📉 DRAWDOWN:")
    print(f"  Max DD        : {max_dd:7.2f}R")
    print(f"  Max DD bars   : {max_dd_bars:4d} trades")
    print(f"  DD % of equity: {abs(max_dd) / float(equity_curve.iloc[-1]) * 100:6.2f}%")

    print(f"\n⚖️ SIDE BREAKDOWN:")
    print(f"  Long trades   : {len(long_trades):4d} (WR: {long_wr * 100:5.1f}%)")
    print(f"  Short trades  : {len(short_trades):4d} (WR: {short_wr * 100:5.1f}%)")

    print(f"\n⏱️ PERFORMANCE:")
    print(f"  Planning      : {planning_time:6.2f}s")
    print(f"  Bars/second   : {total_bars / planning_time:8.1f}")

    print("\n" + "=" * 80 + "\n")

    # Pass config to visualiser
    config = {
        'regime_filter': regime_filter,
        'chop_threshold': chop_threshold,
        'stop_buffer': stop_buffer,
        'cooldown_bars': cooldown_bars,
        'costs_per_trade_r': costs_per_trade_r,
        'daily_sharpe': daily_sharpe_metrics['daily_sharpe'],
        'annual_return': daily_sharpe_metrics['annualized_return'],
        'annual_vol': daily_sharpe_metrics['annualized_vol']
    }

    generate_performance_report(
        results_r,
        equity_curve,
        drawdown,
        symbol=symbol,
        days=days,
        m5_data=m5_data,
        trades=final_trades,
        config=config
    )

    client.shutdown()

    return {
        "days": days,
        "trades": len(res),
        "net_r": float(equity_curve.iloc[-1]),
        "winrate": winrate,
        "sharpe": sharpe,
        "daily_sharpe": daily_sharpe_metrics['daily_sharpe'],
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "avg_r": mean_r,
        "long_wr": long_wr,
        "short_wr": short_wr,
        "expectancy": mean_r,
        "recovery_factor": recovery_factor,
        "mar_ratio": mar_ratio,
        "annual_r": annual_r,
        "regime_filter": regime_filter,
        "choppy_segments_skipped": skipped_choppy if regime_filter else 0,
        "costs_per_trade_r": costs_per_trade_r,
        "annualized_return": daily_sharpe_metrics['annualized_return'],
        "annualized_vol": daily_sharpe_metrics['annualized_vol'],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="US500.cash")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--max-trades-day", type=int, default=2)

    # Strategy params
    parser.add_argument("--min-slope", type=float, default=0.15)
    parser.add_argument("--ema-period", type=int, default=20)
    parser.add_argument("--pullback-bars", type=int, default=3)
    parser.add_argument("--signal-close-frac", type=float, default=0.30)
    parser.add_argument("--stop-buffer", type=float, default=2.0)
    parser.add_argument("--min-risk", type=float, default=2.0)
    parser.add_argument("--cooldown", type=int, default=10)

    # Regime filter
    parser.add_argument("--regime-filter", action="store_true",
                        help="Enable regime filter (skip choppy markets)")
    parser.add_argument("--chop-threshold", type=float, default=2.5,
                        help="Chop threshold (higher = stricter)")

    # Costs
    parser.add_argument("--costs", type=float, default=0.0,
                        help="Trading costs per trade in R (e.g., 0.04 for spread+slippage)")

    args = parser.parse_args()

    run_backtest(
        args.symbol,
        args.days,
        max_trades_day=args.max_trades_day,
        min_slope=args.min_slope,
        ema_period=args.ema_period,
        pullback_bars=args.pullback_bars,
        signal_close_frac=args.signal_close_frac,
        stop_buffer=args.stop_buffer,
        min_risk_price_units=args.min_risk,
        cooldown_bars=args.cooldown,
        regime_filter=args.regime_filter,
        chop_threshold=args.chop_threshold,
        costs_per_trade_r=args.costs,
    )