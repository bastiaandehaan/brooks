# main.py
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import Side, H2L2Params, plan_h2l2_trades
from utils.mt5_client import Mt5Client, Mt5ConnectionParams
from utils.mt5_data import RatesRequest, fetch_rates
from utils.symbol_spec import SymbolSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("brooks")


def main() -> int:
    p = argparse.ArgumentParser(description="Brooks H2/L2 planner (NEXT_OPEN), no execution yet")

    # Market
    p.add_argument("--symbol", default="US500.cash")
    p.add_argument("--m5-bars", type=int, default=500)
    p.add_argument("--m15-bars", type=int, default=300)

    # Guardrails (interpret session and day limits in these timezones)
    p.add_argument("--max-trades-day", type=int, default=2)
    p.add_argument("--session-tz", default="America/New_York")
    p.add_argument("--day-tz", default="America/New_York")
    p.add_argument("--session-start", default="09:30")  # NY cash open
    p.add_argument("--session-end", default="15:00")    # stop new trades earlier than close

    # Strategy params
    p.add_argument("--min-risk", type=float, default=1.0, help="Minimum stop distance in price units (US500 points)")
    p.add_argument("--close-frac", type=float, default=0.25, help="Signal bar close near extreme fraction")
    p.add_argument("--cooldown-bars", type=int, default=3)

    # Trend params
    p.add_argument("--ema", type=int, default=20)
    p.add_argument("--slope-lookback", type=int, default=10)
    p.add_argument("--confirm-bars", type=int, default=20)
    p.add_argument("--min-above-frac", type=float, default=0.60)
    p.add_argument("--min-close-ema-dist", type=float, default=0.50)
    p.add_argument("--min-slope", type=float, default=0.20)
    p.add_argument("--pullback-allowance", type=float, default=2.00)

    args = p.parse_args()

    # Time sanity (laptop time != MT5 server time; we anchor on UTC and convert)
    now_utc = datetime.now(timezone.utc)
    logger.info(
        "Clock UTC=%s Brussels=%s NewYork=%s",
        now_utc.isoformat(timespec="seconds"),
        now_utc.astimezone(ZoneInfo("Europe/Brussels")).isoformat(timespec="seconds"),
        now_utc.astimezone(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
    )

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

        # Fetch M15 and infer trend
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

        # Fetch M5 and plan trades
        m5 = fetch_rates(
            mt5,
            RatesRequest(symbol=args.symbol, timeframe=mt5.TIMEFRAME_M5, count=args.m5_bars),
        )

        strat_params = H2L2Params(
            min_risk_price_units=args.min_risk,
            signal_close_frac=args.close_frac,
            cooldown_bars=args.cooldown_bars,
        )
        plans = plan_h2l2_trades(m5, side, spec, strat_params)
        logger.info("Planner: planned trades=%d (before guardrails)", len(plans))

        # Apply guardrails in NY time (DST-safe)
        guard = Guardrails(
            session_tz=args.session_tz,
            day_tz=args.day_tz,
            session_start=args.session_start,
            session_end=args.session_end,
            max_trades_per_day=args.max_trades_day,
            one_trade_per_execute_ts=True,
        )
        accepted, rejected = apply_guardrails(plans, guard)

        logger.info("After guardrails: accepted=%d rejected=%d", len(accepted), len(rejected))

        for t in accepted[-5:]:
            logger.info(
                "ACCEPT Signal=%s Exec=%s Side=%s Stop=%.2f Reason=%s",
                t.signal_ts, t.execute_ts, t.side, t.stop, t.reason
            )

        for t, reason in rejected[:5]:
            logger.info(
                "REJECT (%s) Signal=%s Exec=%s Side=%s",
                reason, t.signal_ts, t.execute_ts, t.side
            )

        return 0
    finally:
        c.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
