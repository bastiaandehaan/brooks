from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.mt5_client import Mt5Client, Mt5ConnectionParams
from utils.mt5_data import rates_to_df
from utils.symbol_spec import SymbolSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Brooks planner-only (NEXT_OPEN) – no execution.")
    p.add_argument("--symbol", default="US500.cash")
    p.add_argument("--m15-bars", type=int, default=400)
    p.add_argument("--m5-bars", type=int, default=500)
    p.add_argument("--timeframe-minutes", type=int, default=5)
    p.add_argument("--min-risk-points", type=float, default=2.0)

    # Guardrails defaults (bewust conservatief / eenvoudig)
    p.add_argument("--max-trades-per-day", type=int, default=2)
    p.add_argument("--session-start", default="09:30")  # NY time
    p.add_argument("--session-end", default="15:00")    # NY time (entries)
    p.add_argument("--tz", default="America/New_York")
    return p


def _clock_log() -> None:
    now_utc = datetime.now(timezone.utc)
    brussels = now_utc.astimezone(ZoneInfo("Europe/Brussels"))
    newyork = now_utc.astimezone(ZoneInfo("America/New_York"))
    logger.info("Clock UTC=%s Brussels=%s NewYork=%s", now_utc.isoformat(), brussels.isoformat(), newyork.isoformat())


def main() -> None:
    args = build_parser().parse_args()
    _clock_log()

    c = Mt5Client(mt5, Mt5ConnectionParams())
    c.initialize()

    try:
        symbol = args.symbol
        c.ensure_selected(symbol)

        info = c.symbol_info(symbol)
        spec = SymbolSpec.from_mt5_dict(info)

        logger.info(
            "SymbolSpec: %s",
            {
                "name": spec.name,
                "digits": spec.digits,
                "tick_size": spec.tick_size,
                "tick_value": spec.tick_value,
                "usd_per_price_unit_per_lot": spec.usd_per_price_unit_per_lot,
                "volume_min": spec.volume_min,
                "volume_step": spec.volume_step,
                "volume_max": spec.volume_max,
            },
        )

        # --- M15 trend filter ---
        m15_rates = c.copy_rates(symbol, mt5.TIMEFRAME_M15, args.m15_bars)
        m15 = rates_to_df(m15_rates)
        trend, metrics = infer_trend_m15(m15)

        logger.info(
            "Trend(M15) side=%s metrics=%s",
            trend.name,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(metrics).items()},
        )

        # --- M5 for H2/L2 ---
        m5_rates = c.copy_rates(symbol, mt5.TIMEFRAME_M5, args.m5_bars)
        m5 = rates_to_df(m5_rates)

        if len(m5) < 10:
            logger.warning("Not enough M5 bars: n=%d", len(m5))
            return

        last_ts = m5.index[-1]
        now_utc = datetime.now(timezone.utc)
        last_ts_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_sec = (now_utc - last_ts_utc.to_pydatetime()).total_seconds()
        logger.info("M5 last bar ts=%s age_sec=%.0f", last_ts.isoformat(), age_sec)

        params = H2L2Params(min_risk_points=float(args.min_risk_points))

        candidate = plan_next_open_trade(
            m5=m5,
            trend=Side.LONG if trend == Side.LONG else Side.SHORT,
            spec=spec,
            p=params,
            timeframe_minutes=int(args.timeframe_minutes),
            now_utc=now_utc,  # blokkeert fallback als last bar nog forming is
        )

        if candidate is None:
            logger.info("Planner: no NEXT_OPEN candidate for last bar.")
            return

        guard = Guardrails(
            timezone=args.tz,
            session_start=args.session_start,
            session_end=args.session_end,
            max_trades_per_day=int(args.max_trades_per_day),
        )

        accepted, rejected = apply_guardrails([candidate], guard)

        logger.info("NEXT_OPEN candidate: %s", asdict(candidate))
        logger.info("After guardrails: accepted=%d rejected=%d", len(accepted), len(rejected))

        for t in accepted:
            logger.info("ACCEPT trade: %s", asdict(t))
        for t, reason in rejected:
            logger.info("REJECT trade: %s reason=%s", asdict(t), reason)

    finally:
        c.shutdown()


if __name__ == "__main__":
    main()
