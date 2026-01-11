# backtest/runner.py
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.mt5_client import Mt5Client, Mt5ConnectionParams
from utils.mt5_data import RatesRequest, fetch_rates
from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    accepted: list
    rejected: list


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--symbol", default="US500.cash")
    p.add_argument("--timeframe-minutes", type=int, default=5)
    p.add_argument("--m15-bars", type=int, default=300)
    p.add_argument("--m5-bars", type=int, default=500)

    # Trend params
    p.add_argument("--ema", type=int, default=20)
    p.add_argument("--slope-lookback", type=int, default=10)
    p.add_argument("--confirm-bars", type=int, default=20)
    p.add_argument("--min-above-frac", type=float, default=0.60)
    p.add_argument("--min-close-ema-dist", type=float, default=0.50)
    p.add_argument("--min-slope", type=float, default=0.20)
    p.add_argument("--pullback-allowance", type=float, default=2.00)

    # Strategy params
    p.add_argument(
        "--min-risk-price-units",
        "--min-risk-points",
        dest="min_risk_price_units",
        type=float,
        default=1.0,
    )
    p.add_argument("--signal-close-frac", type=float, default=0.25)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--pullback-bars", type=int, default=0)

    # Guardrails
    p.add_argument("--session-tz", dest="session_tz", default="America/New_York")
    p.add_argument("--day-tz", dest="day_tz", default="America/New_York")
    p.add_argument("--session-start", default="09:30")
    p.add_argument("--session-end", default="15:00")
    p.add_argument(
        "--max-trades-day",
        "--max-trades-per-day",
        dest="max_trades_day",
        type=int,
        default=2,
    )

    p.add_argument("--log-level", default="INFO")

    return p


def _map_timeframe_minutes(mt5, minutes: int) -> int:
    mapping = {
        1: mt5.TIMEFRAME_M1,
        5: mt5.TIMEFRAME_M5,
        15: mt5.TIMEFRAME_M15,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
        240: mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
    }
    if minutes not in mapping:
        raise ValueError(f"Unsupported timeframe_minutes={minutes}. Supported={sorted(mapping)}")
    return mapping[minutes]


def _append_synthetic_current_bar(m5: pd.DataFrame, timeframe_minutes: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    last_ts = m5.index[-1]
    current_ts = last_ts + pd.Timedelta(minutes=timeframe_minutes)
    if current_ts in m5.index:
        return m5, current_ts

    syn = pd.DataFrame(
        {"open": [float("nan")], "high": [float("nan")], "low": [float("nan")], "close": [float("nan")]},
        index=pd.DatetimeIndex([current_ts]),
    )
    syn.index = syn.index.tz_convert(last_ts.tz)
    return pd.concat([m5, syn], axis=0), current_ts


def _iter_days(m5: pd.DataFrame, day_tz: str) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    tz = ZoneInfo(day_tz)
    day_index = m5.index.tz_convert(tz).normalize()
    for day, group in m5.groupby(day_index, sort=True):
        yield day, group


def run_backtest(
    m5: pd.DataFrame,
    *,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
    guardrails: Guardrails,
    timeframe_minutes: int,
) -> BacktestResult:
    if m5.empty:
        raise ValueError("m5 data empty")

    if m5.index.tz is None:
        raise ValueError("m5 index must be tz-aware")

    m5 = m5.sort_index()
    m5 = m5[~m5.index.duplicated(keep="last")]

    accepted: list = []
    rejected: list = []

    for day, day_bars in _iter_days(m5, guardrails.day_tz):
        logger.info("Backtest day=%s bars=%d", day.date(), len(day_bars))
        for ts in day_bars.index:
            history = m5.loc[:ts]
            if len(history) < 3:
                continue

            history_current, current_ts = _append_synthetic_current_bar(history, timeframe_minutes)
            candidate = plan_next_open_trade(
                history_current,
                trend=trend,
                spec=spec,
                p=p,
                timeframe_minutes=timeframe_minutes,
                now_utc=current_ts,
            )

            if candidate is None:
                continue

            if candidate.execute_ts != current_ts:
                logger.debug(
                    "Skipping candidate with execute_ts=%s (current=%s)",
                    candidate.execute_ts,
                    current_ts,
                )
                continue

            planned = [*accepted, candidate]
            accepted_all, rejected_all = apply_guardrails(planned, guardrails)
            if len(accepted_all) > len(accepted):
                accepted = accepted_all
                logger.info(
                    "ACCEPT: side=%s signal=%s exec=%s stop=%.5f reason=%s",
                    candidate.side,
                    candidate.signal_ts,
                    candidate.execute_ts,
                    candidate.stop,
                    candidate.reason,
                )
            else:
                reason = next((r for t, r in rejected_all if t == candidate), "guardrail reject")
                rejected.append((candidate, reason))
                logger.info(
                    "REJECT: side=%s signal=%s exec=%s reason=%s",
                    candidate.side,
                    candidate.signal_ts,
                    candidate.execute_ts,
                    reason,
                )

    return BacktestResult(accepted=accepted, rejected=rejected)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    now_utc = datetime.now(timezone.utc)
    logger.info(
        "Clock UTC=%s Brussels=%s NewYork=%s",
        now_utc.isoformat(timespec="seconds"),
        now_utc.astimezone(ZoneInfo("Europe/Brussels")).isoformat(timespec="seconds"),
        now_utc.astimezone(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
    )

    import MetaTrader5 as mt5  # local import keeps tests lighter

    c = Mt5Client(mt5, Mt5ConnectionParams())
    c.initialize()
    try:
        c.ensure_selected(args.symbol)
        info = c.symbol_info(args.symbol)
        spec = SymbolSpec.from_symbol_info(info)
        logger.info(
            "SymbolSpec: %s digits=%d tick_size=%.5f tick_value=%.5f usd/price_unit/lot=%.5f vol_min=%.2f step=%.2f",
            spec.name,
            spec.digits,
            spec.tick_size,
            spec.tick_value,
            spec.usd_per_price_unit_per_lot,
            spec.volume_min,
            spec.volume_step,
        )

        m15 = fetch_rates(
            mt5,
            RatesRequest(symbol=args.symbol, timeframe=mt5.TIMEFRAME_M15, count=args.m15_bars),
        )
        trend_params = TrendParams(
            ema_period=args.ema,
            slope_lookback=args.slope_lookback,
            confirm_bars=args.confirm_bars,
            min_above_frac=args.min_above_frac,
            min_close_ema_dist=args.min_close_ema_dist,
            min_slope=args.min_slope,
            pullback_allowance=args.pullback_allowance,
        )
        trend, metrics = infer_trend_m15(m15, trend_params)
        logger.info(
            "Trend metrics: close=%.2f ema=%.2f d=%.2f close-ema=%.2f slope=%.3f above=%.2f below=%.2f",
            metrics.last_close,
            metrics.last_ema,
            metrics.close_ema_dist,
            metrics.last_close_minus_ema,
            metrics.ema_slope,
            metrics.above_frac,
            metrics.below_frac,
        )

        if trend is None:
            logger.info("Trend filter: NONE (skip)")
            return 0

        side = Side.LONG if trend == Trend.BULL else Side.SHORT
        logger.info("Trend filter: %s => %s", trend, side)

        timeframe = _map_timeframe_minutes(mt5, args.timeframe_minutes)
        m5 = fetch_rates(
            mt5,
            RatesRequest(symbol=args.symbol, timeframe=timeframe, count=args.m5_bars),
        )

        strat_params = H2L2Params(
            min_risk_price_units=args.min_risk_price_units,
            signal_close_frac=args.signal_close_frac,
            cooldown_bars=args.cooldown_bars,
            pullback_bars=args.pullback_bars,
        )

        guardrails = Guardrails(
            session_tz=args.session_tz,
            day_tz=args.day_tz,
            session_start=args.session_start,
            session_end=args.session_end,
            max_trades_per_day=args.max_trades_day,
        )

        result = run_backtest(
            m5,
            trend=side,
            spec=spec,
            p=strat_params,
            guardrails=guardrails,
            timeframe_minutes=args.timeframe_minutes,
        )

        logger.info("Backtest summary: accepted=%d rejected=%d", len(result.accepted), len(result.rejected))
        return 0

    finally:
        c.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
