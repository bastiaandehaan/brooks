#!/usr/bin/env python3
"""
Brooks Live Monitor - Production-safe monitor (manual execution workflow)

Key rules:
- Session window is NYSE cash hours: 09:30-16:00 ET
- Optional "no-new-trades" cutoff before the close (default 15:30 ET)
  to avoid late-session noise/whipsaws.
- This script only DETECTS and NOTIFIES. No order execution.

Usage example:
python scripts/live_monitor.py --symbol US500.cash --risk-pct 0.5 --regime-filter --chop-threshold 2.0 --stop-buffer 1.0 --interval 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass

import MetaTrader5 as mt5
import pandas as pd

# project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.ftmo_guardian import FTMOAccountType, FTMOGuardian
from execution.guardrails import Guardrails, apply_guardrails
from execution.risk_manager import RiskManager
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from strategies.regime import RegimeParams, should_trade_today
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates
from utils.telegram_bot import TelegramBot, TradingSignal

# Optional debug logger (best-effort)
try:
    from utils.debug_logger import DebugLogger, capture_error_context  # type: ignore

    DEBUG_AVAILABLE = True
except Exception:
    DEBUG_AVAILABLE = False
    DebugLogger = None  # type: ignore


logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


@dataclass(frozen=True)
class SessionConfig:
    tz: str = NY_TZ
    session_start: str = "09:30"  # ET
    session_end: str = "16:00"  # ET (NYSE cash close)
    trade_cutoff: str = "15:30"  # ET (no new trades after this time)


def _parse_hhmm(hhmm: str) -> pd.Timestamp:
    # Only used for .time() extraction; date part irrelevant.
    return pd.Timestamp(hhmm)


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def session_state(cfg: SessionConfig, ts: pd.Timestamp | None = None) -> tuple[str, pd.Timestamp]:
    """
    Returns (state, now_ny_ts) where state ∈ {"OUTSIDE", "ACTIVE", "CUTOFF"}.

    OUTSIDE: outside 09:30-16:00 ET
    ACTIVE : in session and before cutoff
    CUTOFF : in session but after cutoff (no new trades)
    """
    ts_ny = ts if ts is not None else now_ny()
    cur = ts_ny.time()

    start = _parse_hhmm(cfg.session_start).time()
    end = _parse_hhmm(cfg.session_end).time()
    cutoff = _parse_hhmm(cfg.trade_cutoff).time()

    if cur < start or cur > end:
        return "OUTSIDE", ts_ny
    if cur >= cutoff:
        return "CUTOFF", ts_ny
    return "ACTIVE", ts_ny


def check_emergency_stop(project_root: str | None = None) -> tuple[bool, str | None]:
    """STOP.txt in project root stops the monitor."""
    root = project_root or os.getcwd()
    stop_file = os.path.join(root, "STOP.txt")
    if os.path.exists(stop_file):
        try:
            reason = open(stop_file, encoding="utf-8", errors="ignore").read().strip()
            return True, reason if reason else "Emergency stop activated"
        except Exception:
            return True, "Emergency stop file found"
    return False, None


def _hygiene(df: pd.DataFrame) -> pd.DataFrame:
    # keep it simple: sort index, drop duplicate timestamps
    if df is None or df.empty:
        return df
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def check_for_signals(
    *,
    symbol: str,
    risk_pct: float,
    regime_filter: bool,
    chop_threshold: float,
    stop_buffer: float,
    ftmo_guardian: FTMOGuardian | None,
    telegram_bot: TelegramBot,
    debug_logger: DebugLogger | None,
) -> bool:
    """
    Check for Brooks signals and send Telegram notification if found.
    Returns True if a signal was sent.
    """
    logger.info("🔍 Checking for signals...")

    m15_data: pd.DataFrame | None = None
    m5_data: pd.DataFrame | None = None

    config = {
        "symbol": symbol,
        "risk_pct": risk_pct,
        "regime_filter": regime_filter,
        "chop_threshold": chop_threshold,
        "stop_buffer": stop_buffer,
    }

    try:
        client = Mt5Client(mt5_module=mt5)
        if not client.initialize():
            logger.error("❌ Failed to connect to MT5")
            telegram_bot.send_error("MT5 connection failed")
            return False

        spec = client.get_symbol_specification(symbol)
        if spec is None:
            logger.error("❌ Symbol %s not found", symbol)
            client.shutdown()
            return False

        # Fetch data
        req_m15 = RatesRequest(symbol, mt5.TIMEFRAME_M15, 300)
        req_m5 = RatesRequest(symbol, mt5.TIMEFRAME_M5, 500)

        m15_data = _hygiene(fetch_rates(mt5, req_m15))
        m5_data = _hygiene(fetch_rates(mt5, req_m5))

        if m15_data.empty or m5_data.empty:
            logger.warning(
                "⚠️ Empty dataframes from MT5 (m15=%s, m5=%s)", len(m15_data), len(m5_data)
            )
            client.shutdown()
            return False

        # Regime filter
        regime_status = "UNKNOWN"
        if regime_filter:
            regime_params = RegimeParams(chop_threshold=chop_threshold)
            ok, reason = should_trade_today(m15_data, regime_params)
            if not ok:
                logger.info("⛔ Regime filter: %s", reason)
                client.shutdown()
                return False
            logger.info("✅ Regime filter: %s", reason)
            regime_status = "TRENDING"

        # FTMO check (optional)
        if ftmo_guardian:
            acc = mt5.account_info()
            if acc:
                can_trade, limit_reason = ftmo_guardian.can_open_trade(acc.balance)
                if not can_trade:
                    logger.warning("⛔ FTMO Guardian: %s", limit_reason)
                    telegram_bot.send_error(f"FTMO limit: {limit_reason}")
                    client.shutdown()
                    return False

        # Trend inference (M15)
        tparams = TrendParams()
        trend, trend_reason = infer_trend_m15(m15_data, tparams)
        logger.info(
            "Trend: %s (%s)", trend.value if hasattr(trend, "value") else str(trend), trend_reason
        )

        if trend not in (Trend.BULL, Trend.BEAR):
            logger.info("No clear trend")
            client.shutdown()
            return False

        side = Side.LONG if trend == Trend.BULL else Side.SHORT

        # Plan trade (NEXT_OPEN contract)
        hparams = H2L2Params(
            min_risk_price_units=1.0, signal_close_frac=0.30, pullback_bars=2, cooldown_bars=0
        )
        planned = plan_next_open_trade(m5_data, side, spec, hparams, timeframe_minutes=5)

        if not planned:
            logger.info("No setup")
            client.shutdown()
            return False

        # Risk sizing
        rm = RiskManager()
        lots, risk_usd = rm.size_position(
            balance=float(mt5.account_info().balance) if mt5.account_info() else 0.0,
            risk_pct=risk_pct,
            entry=planned.entry,
            stop=planned.stop,
            spec=spec,
            min_risk_price_units=1.0,
            fees_usd=0.0,
        )

        if lots <= 0:
            logger.info("Sizing rejected (lots=%s risk_usd=%s)", lots, risk_usd)
            client.shutdown()
            return False

        # Guardrails (1 trade per timestamp etc. happens in your guardrails)
        g = Guardrails(
            max_trades_per_day=2,
            session_start="09:30",
            session_end="16:00",
            day_tz=NY_TZ,
            session_tz=NY_TZ,
        )

        accepted, rejected = apply_guardrails([planned], g)
        if not accepted:
            logger.info(
                "Guardrails rejected trade (reason=%s)",
                rejected[0].reason if rejected else "unknown",
            )
            client.shutdown()
            return False

        pick = accepted[0]

        # Send notification
        sig = TradingSignal(
            symbol=symbol,
            side=pick.side.value,
            entry=pick.entry,
            stop=pick.stop,
            target=pick.tp,
            reason=pick.reason,
            timeframe="M5",
        )
        telegram_bot.send_signal(sig, lots=lots, risk_usd=risk_usd, regime=regime_status)

        # Debug dump
        if DEBUG_AVAILABLE and debug_logger:
            try:
                debug_logger.log_trade(
                    {
                        "ts": now_ny().isoformat(),
                        "symbol": symbol,
                        "side": pick.side.value,
                        "entry": pick.entry,
                        "stop": pick.stop,
                        "target": pick.tp,
                        "lots": lots,
                        "risk_usd": risk_usd,
                        "risk_pct": risk_pct,
                        "reason": pick.reason,
                        "regime": regime_status,
                        "status": "signaled",
                    }
                )
            except Exception:
                pass

        client.shutdown()
        return True

    except Exception as e:
        logger.error("❌ Error checking signals: %s", e, exc_info=True)

        if DEBUG_AVAILABLE and debug_logger:
            try:
                ctx = capture_error_context(e, market_data=m5_data, config=config)  # type: ignore
                debug_logger.log_error(ctx)  # type: ignore
            except Exception:
                pass

        try:
            telegram_bot.send_error(f"Error: {str(e)}")
        except Exception:
            pass

        return False


def run_monitor(
    *,
    symbol: str,
    risk_pct: float,
    regime_filter: bool,
    chop_threshold: float,
    stop_buffer: float,
    check_interval: int,
    enable_ftmo_protection: bool,
    ftmo_account_size: int,
    cfg: SessionConfig,
) -> None:
    # Debug logger
    debug_logger = None
    if DEBUG_AVAILABLE:
        try:
            debug_logger = DebugLogger(log_dir="logs")  # type: ignore
            logger.info("✅ Debug logging enabled")
        except Exception as e:
            logger.warning("⚠️ Could not initialize debug logger: %s", e)

    telegram_bot = TelegramBot()

    # FTMO Guardian (optional)
    ftmo_guardian: FTMOGuardian | None = None
    if enable_ftmo_protection:
        try:
            account_type = (
                FTMOAccountType.CHALLENGE_10K
                if int(ftmo_account_size) == 10000
                else FTMOAccountType.CHALLENGE_25K
            )
            ftmo_guardian = FTMOGuardian(account_type=account_type)
            logger.info("✅ FTMO Guardian enabled")
        except Exception as e:
            logger.error("❌ Could not initialize FTMO Guardian: %s", e)
            logger.info("Continuing without FTMO protection.")

    startup_msg = (
        "🤖 <b>Brooks Live Monitor Started</b>\n\n"
        f"Symbol: {symbol}\n"
        f"Risk: {risk_pct}%\n"
        f"Regime Filter: {'ON' if regime_filter else 'OFF'}\n"
        f"Check Interval: {check_interval}s\n"
        f"Session: {cfg.session_start}-{cfg.session_end} ET\n"
        f"No-new-trades after: {cfg.trade_cutoff} ET\n"
        f"FTMO Protection: {'ENABLED' if ftmo_guardian else 'DISABLED'}\n"
        f"Debug Logging: {'ON' if debug_logger else 'OFF'}\n"
    )
    telegram_bot.send_message(startup_msg)

    iteration = 0
    try:
        while True:
            iteration += 1

            stop_requested, stop_reason = check_emergency_stop()
            if stop_requested:
                msg = f"🛑 Emergency stop: {stop_reason}"
                logger.warning(msg)
                telegram_bot.send_error(msg)
                break

            state, ts_ny = session_state(cfg)
            logger.info(
                "NY time now: %s ET | state=%s | check=%d",
                ts_ny.strftime("%H:%M:%S"),
                state,
                iteration,
            )

            if state == "OUTSIDE":
                logger.info("⏸️ Outside NY session - sleeping %ss...", check_interval)
                time.sleep(check_interval)
                continue

            if state == "CUTOFF":
                # This is the key “extra”: stay alive, but do nothing new late-session.
                logger.info(
                    "🟠 Cutoff window active (no new trades). Sleeping %ss...", check_interval
                )
                time.sleep(check_interval)
                continue

            # ACTIVE
            check_for_signals(
                symbol=symbol,
                risk_pct=risk_pct,
                regime_filter=regime_filter,
                chop_threshold=chop_threshold,
                stop_buffer=stop_buffer,
                ftmo_guardian=ftmo_guardian,
                telegram_bot=telegram_bot,
                debug_logger=debug_logger,
            )

            logger.info("💤 Sleeping %ss...", check_interval)
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("⛔ Monitor stopped by user")
        telegram_bot.send_message("⛔ <b>Brooks Live Monitor Stopped</b>")
    except Exception as e:
        logger.error("❌ Monitor crashed: %s", e, exc_info=True)
        telegram_bot.send_error(f"Monitor crashed: {str(e)}")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Brooks Live Signal Monitor (manual execution)")
    parser.add_argument("--symbol", default="US500.cash")
    parser.add_argument("--risk-pct", type=float, default=0.5)
    parser.add_argument("--regime-filter", action="store_true")
    parser.add_argument("--chop-threshold", type=float, default=2.0)
    parser.add_argument("--stop-buffer", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=30, help="Seconds between checks")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    # Session config overrides
    parser.add_argument("--session-start", default="09:30")
    parser.add_argument("--session-end", default="16:00")
    parser.add_argument("--trade-cutoff", default="15:30", help="No new trades after this ET time")

    # FTMO
    parser.add_argument("--ftmo-protection", action="store_true", default=False)
    parser.add_argument("--ftmo-account-size", type=int, default=10000, choices=[10000, 25000])

    args = parser.parse_args()
    _setup_logging(args.log_level)

    cfg = SessionConfig(
        session_start=args.session_start,
        session_end=args.session_end,
        trade_cutoff=args.trade_cutoff,
    )

    logger.info("=" * 60)
    logger.info("🤖 BROOKS LIVE MONITOR STARTING")
    logger.info("=" * 60)
    logger.info("Symbol           : %s", args.symbol)
    logger.info("Risk per trade   : %s%%", args.risk_pct)
    logger.info("Regime filter    : %s", "ENABLED" if args.regime_filter else "DISABLED")
    logger.info("Chop threshold   : %s", args.chop_threshold)
    logger.info("Stop buffer      : %s", args.stop_buffer)
    logger.info("Check interval   : %ss", args.interval)
    logger.info("NY Session hours : %s-%s ET", cfg.session_start, cfg.session_end)
    logger.info("No-new-trades    : after %s ET", cfg.trade_cutoff)
    logger.info("FTMO Protection  : %s", "ENABLED" if args.ftmo_protection else "DISABLED")
    logger.info("=" * 60)

    run_monitor(
        symbol=args.symbol,
        risk_pct=args.risk_pct,
        regime_filter=args.regime_filter,
        chop_threshold=args.chop_threshold,
        stop_buffer=args.stop_buffer,
        check_interval=args.interval,
        enable_ftmo_protection=args.ftmo_protection,
        ftmo_account_size=args.ftmo_account_size,
        cfg=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
