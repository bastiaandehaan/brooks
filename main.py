# main.py
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.mt5_client import Mt5Client, Mt5ConnectionParams
from utils.mt5_data import RatesRequest, fetch_rates
from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--symbol", default="US500.cash")
    p.add_argument("--m15-bars", type=int, default=300)
    p.add_argument("--m5-bars", type=int, default=500)
    p.add_argument("--timeframe-minutes", type=int, default=5)

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # Log clocks (helpful when debugging session filters)
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
        # Symbol + spec
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

        # M15 trend
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

        # M5 data
        m5 = fetch_rates(
            mt5,
            RatesRequest(symbol=args.symbol, timeframe=mt5.TIMEFRAME_M5, count=args.m5_bars),
        )
        last_bar_ts = m5.index[-1]
        logger.info("M5 last bar ts: %s", last_bar_ts)

        strat_params = H2L2Params(
            min_risk_price_units=args.min_risk_price_units,
            signal_close_frac=args.signal_close_frac,
            cooldown_bars=args.cooldown_bars,
        )

        t = plan_next_open_trade(
            m5,
            trend=side,
            spec=spec,
            p=strat_params,
            timeframe_minutes=args.timeframe_minutes,
            now_utc=pd.Timestamp.now(tz="UTC"),
        )

        if t is None:
            logger.info("Planner: no NEXT_OPEN candidate.")
            return 0

        g = Guardrails(
            session_tz=args.session_tz,
            day_tz=args.day_tz,
            session_start=args.session_start,
            session_end=args.session_end,
            max_trades_per_day=args.max_trades_day,
        )
        accepted, rejected = apply_guardrails([t], g)

        if rejected:
            for _, reason in rejected:
                logger.info("Guardrail reject: %s", reason)

        if not accepted:
            logger.info("Planner: candidate rejected by guardrails.")
            return 0

        pick = accepted[0]
        logger.info(
            "CANDIDATE: side=%s signal=%s exec=%s stop=%.5f reason=%s",
            pick.side,
            pick.signal_ts,
            pick.execute_ts,
            pick.stop,
            pick.reason,
        )
        return 0

    finally:
        c.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
