#!/usr/bin/env python3
"""
Brooks Live Monitor - Production Version with Debug Logging
"""
import sys
import os
import time
import logging
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from strategies.regime import RegimeParams, should_trade_today
from execution.guardrails import Guardrails, apply_guardrails
from execution.risk_manager import RiskManager
from execution.ftmo_guardian import FTMOGuardian, FTMOAccountType
from utils.telegram_bot import TelegramBot, TradingSignal

# Import debug logger (with fallback if not available)
try:
    from utils.debug_logger import DebugLogger, capture_error_context

    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    print("⚠️ Debug logger not available - errors won't be logged to files")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "15:00"
CHECK_INTERVAL = 300


def is_ny_session_active() -> bool:
    """Check if we're currently in NY session"""
    now_ny = pd.Timestamp.now(tz=NY_TZ)
    current_time = now_ny.time()
    start = pd.Timestamp(SESSION_START).time()
    end = pd.Timestamp(SESSION_END).time()
    return start <= current_time <= end


def check_emergency_stop() -> tuple[bool, Optional[str]]:
    """Check if emergency stop file exists"""
    stop_file = "STOP.txt"
    if os.path.exists(stop_file):
        try:
            with open(stop_file, 'r') as f:
                reason = f.read().strip()
            return True, reason if reason else "Emergency stop activated"
        except Exception:
            return True, "Emergency stop file found"
    return False, None


def check_for_signals(
        symbol: str,
        risk_pct: float,
        regime_filter: bool,
        chop_threshold: float,
        stop_buffer: float,
        ftmo_guardian: Optional[FTMOGuardian],
        telegram_bot: TelegramBot,
        debug_logger: Optional[DebugLogger]
) -> bool:
    """Check for trading signals"""

    logger.info("🔍 Checking for signals...")

    # Store data for debug logging if error occurs
    m15_data = None
    m5_data = None
    config = {
        "symbol": symbol,
        "risk_pct": risk_pct,
        "regime_filter": regime_filter,
        "chop_threshold": chop_threshold,
        "stop_buffer": stop_buffer
    }

    try:
        client = Mt5Client(mt5_module=mt5)
        if not client.initialize():
            logger.error("❌ Failed to connect to MT5")
            telegram_bot.send_error("MT5 connection failed")
            return False

        spec = client.get_symbol_specification(symbol)
        if spec is None:
            logger.error(f"❌ Symbol {symbol} not found")
            client.shutdown()
            return False

        # Fetch data
        req_m15 = RatesRequest(symbol, mt5.TIMEFRAME_M15, 300)
        req_m5 = RatesRequest(symbol, mt5.TIMEFRAME_M5, 500)

        try:
            m15_data = fetch_rates(mt5, req_m15)
            m5_data = fetch_rates(mt5, req_m5)
        except Exception as e:
            logger.error(f"❌ Failed to fetch data: {e}")
            telegram_bot.send_error(f"Data fetch failed: {str(e)}")

            # Log error with debug logger
            if DEBUG_AVAILABLE and debug_logger:
                context = capture_error_context(e, config=config)
                debug_logger.log_error(context)

            client.shutdown()
            return False

        if m15_data.empty or m5_data.empty:
            logger.warning("⚠️ Empty dataframes")
            client.shutdown()
            return False

        # Check regime
        if regime_filter:
            regime_params = RegimeParams(chop_threshold=chop_threshold)
            should_trade, regime_reason = should_trade_today(m15_data, regime_params)

            if not should_trade:
                logger.info(f"⛔ Regime filter: {regime_reason}")
                client.shutdown()
                return False

            logger.info(f"✅ Regime filter: {regime_reason}")

        # Check FTMO
        if ftmo_guardian:
            acc_info = mt5.account_info()
            if acc_info:
                can_trade, limit_reason = ftmo_guardian.can_open_trade(acc_info.balance)
                if not can_trade:
                    logger.warning(f"⛔ FTMO Guardian: {limit_reason}")
                    telegram_bot.send_error(f"FTMO limit: {limit_reason}")
                    client.shutdown()
                    return False

        # Infer trend
        t_params = TrendParams(ema_period=20)
        trend_res, metrics = infer_trend_m15(m15_data, t_params)

        logger.info(
            f"Trend: {trend_res} (close={metrics.last_close:.2f} "
            f"ema={metrics.last_ema:.2f} slope={metrics.ema_slope:.2f})"
        )

        if trend_res not in [Trend.BULL, Trend.BEAR]:
            logger.info("No clear trend")
            client.shutdown()
            return False

        side = Side.LONG if trend_res == Trend.BULL else Side.SHORT

        # Plan trade
        strat_params = H2L2Params(
            pullback_bars=3,
            signal_close_frac=0.30,
            min_risk_price_units=2.0,
            stop_buffer=stop_buffer,
            cooldown_bars=0,
        )

        planned_trade = plan_next_open_trade(
            m5_data,
            trend=side,
            spec=spec,
            p=strat_params,
            timeframe_minutes=5,
            now_utc=pd.Timestamp.now(tz="UTC"),
        )

        if planned_trade is None:
            logger.info("No setup found")
            client.shutdown()
            return False

        # Risk management
        acc_info = mt5.account_info()
        if acc_info is None:
            logger.error("❌ Could not fetch account info")
            client.shutdown()
            return False

        rm = RiskManager(risk_per_trade_pct=risk_pct)
        lots = rm.calculate_lot_size(
            balance=acc_info.balance,
            spec=spec,
            entry=planned_trade.entry,
            stop=planned_trade.stop
        )

        # Guardrails
        g = Guardrails(
            session_tz=NY_TZ,
            day_tz=NY_TZ,
            session_start=SESSION_START,
            session_end=SESSION_END,
            max_trades_per_day=2,
        )

        accepted, rejected = apply_guardrails([planned_trade], g)

        if rejected:
            for _, reason in rejected:
                logger.info(f"Guardrail reject: {reason}")
            client.shutdown()
            return False

        if not accepted:
            logger.info("No valid trades after guardrails")
            client.shutdown()
            return False

        # Signal found!
        pick = accepted[0]
        risk_usd = acc_info.balance * risk_pct / 100

        signal = TradingSignal(
            symbol=symbol,
            side=pick.side.value,
            entry=pick.entry,
            stop=pick.stop,
            tp=pick.tp,
            lots=lots,
            risk_usd=risk_usd,
            reason=pick.reason
        )

        telegram_bot.send_signal(signal)
        logger.info(f"✅ Signal sent: {pick.side} @ {pick.entry:.2f}")

        # Log signal to trades log
        if DEBUG_AVAILABLE and debug_logger:
            signal_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "side": pick.side.value,
                "entry": pick.entry,
                "stop": pick.stop,
                "tp": pick.tp,
                "lots": lots,
                "risk_usd": risk_usd,
                "reason": pick.reason,
                "status": "signaled"
            }
            debug_logger.log_trade(signal_data)

        client.shutdown()
        return True

    except Exception as e:
        logger.error(f"❌ Error checking signals: {e}", exc_info=True)

        # Log full error context
        if DEBUG_AVAILABLE and debug_logger:
            try:
                context = capture_error_context(
                    e,
                    market_data=m5_data,
                    config=config
                )
                debug_logger.log_error(context)
                logger.info(f"📝 Error logged to: logs/errors/")
            except Exception as log_err:
                logger.error(f"Failed to log error: {log_err}")

        try:
            telegram_bot.send_error(f"Error: {str(e)}")
        except Exception:
            pass

        return False


def run_monitor(
        symbol: str,
        risk_pct: float,
        regime_filter: bool,
        chop_threshold: float,
        stop_buffer: float,
        check_interval: int,
        enable_ftmo_protection: bool,
        ftmo_account_size: int
) -> None:
    """Main monitoring loop"""

    # Initialize debug logger
    debug_logger = None
    if DEBUG_AVAILABLE:
        try:
            debug_logger = DebugLogger(log_dir="logs")
            logger.info("✅ Debug logging enabled")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize debug logger: {e}")

    # Initialize components
    telegram_bot = TelegramBot()

    ftmo_guardian: Optional[FTMOGuardian] = None
    if enable_ftmo_protection:
        account_type = FTMOAccountType.CHALLENGE_10K if ftmo_account_size == 10000 else FTMOAccountType.CHALLENGE_25K
        ftmo_guardian = FTMOGuardian(account_type=account_type)
        logger.info("FTMO Guardian initialized:")
        logger.info(f"  Account Type: {ftmo_account_size // 1000}k")
        logger.info(f"  Initial Balance: ${ftmo_guardian.rules.initial_balance:,.2f}")
        logger.info(f"  Max Daily Loss: ${ftmo_guardian.rules.max_daily_loss:,.2f}")
        logger.info(f"  Safe Daily Loss: ${ftmo_guardian.safe_daily_loss:,.2f}")
        logger.info(f"  Max Total Loss: ${ftmo_guardian.rules.max_total_loss:,.2f}")
        logger.info(f"  Safe Total Loss: ${ftmo_guardian.safe_total_loss:,.2f}")

    # Startup notification
    startup_msg = (
        "🤖 <b>Brooks Live Monitor Started</b>\n\n"
        f"Symbol: {symbol}\n"
        f"Risk: {risk_pct}%\n"
        f"Regime Filter: {'ON' if regime_filter else 'OFF'}\n"
        f"Debug Logging: {'ON' if debug_logger else 'OFF'}\n"
    )
    if ftmo_guardian:
        startup_msg += f"FTMO Protection: ENABLED ({ftmo_account_size // 1000}k)\n"

    telegram_bot.send_message(startup_msg)
    logger.info("✅ FTMO Guardian enabled" if ftmo_guardian else "⚠️ FTMO Guardian disabled")

    # Main loop
    while True:
        try:
            # Emergency stop check
            stop_requested, stop_reason = check_emergency_stop()
            if stop_requested:
                msg = f"🛑 Emergency stop: {stop_reason}"
                logger.warning(msg)
                telegram_bot.send_error(msg)
                break

            # Session check
            if not is_ny_session_active():
                logger.info(f"⏸️ Outside NY session - sleeping {check_interval}s...")
                time.sleep(check_interval)
                continue

            # Check for signals
            check_for_signals(
                symbol=symbol,
                risk_pct=risk_pct,
                regime_filter=regime_filter,
                chop_threshold=chop_threshold,
                stop_buffer=stop_buffer,
                ftmo_guardian=ftmo_guardian,
                telegram_bot=telegram_bot,
                debug_logger=debug_logger
            )

            logger.info(f"💤 Sleeping {check_interval}s...")
            time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("⚠️ Keyboard interrupt - stopping")
            telegram_bot.send_message("🛑 Monitor stopped manually")
            break
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}", exc_info=True)

            # Log critical error
            if DEBUG_AVAILABLE and debug_logger:
                try:
                    context = capture_error_context(e, config={"critical": "main_loop_error"})
                    debug_logger.log_error(context)
                except Exception:
                    pass

            telegram_bot.send_error(f"Monitor error: {str(e)}")
            time.sleep(60)


def main() -> None:
    """Entry point"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="US500.cash")
    parser.add_argument("--risk-pct", type=float, default=0.5)
    parser.add_argument("--regime-filter", action="store_true")
    parser.add_argument("--chop-threshold", type=float, default=2.0)
    parser.add_argument("--stop-buffer", type=float, default=1.0)
    parser.add_argument("--check-interval", type=int, default=300)
    parser.add_argument("--ftmo-protection", action="store_true")
    parser.add_argument("--ftmo-account-size", type=int, default=10000, choices=[10000, 25000])

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🤖 BROOKS LIVE MONITOR STARTING")
    logger.info("=" * 60)
    logger.info(f"Symbol           : {args.symbol}")
    logger.info(f"Risk per trade   : {args.risk_pct}%")
    logger.info(f"Regime filter    : {'ENABLED' if args.regime_filter else 'DISABLED'}")
    if args.regime_filter:
        logger.info(f"Chop threshold   : {args.chop_threshold}")
    logger.info(f"Stop buffer      : {args.stop_buffer}")
    logger.info(f"Check interval   : {args.check_interval}s ({args.check_interval / 60:.1f} min)")
    logger.info(f"NY Session hours : {SESSION_START}-{SESSION_END} EST")
    logger.info(f"FTMO Protection  : {'ENABLED' if args.ftmo_protection else 'DISABLED'}")
    logger.info("=" * 60)

    # Test Telegram
    try:
        telegram_bot = TelegramBot()
        logger.info("✅ Telegram bot connected")
    except Exception as e:
        logger.error(f"❌ Telegram failed: {e}")
        return

    run_monitor(
        symbol=args.symbol,
        risk_pct=args.risk_pct,
        regime_filter=args.regime_filter,
        chop_threshold=args.chop_threshold,
        stop_buffer=args.stop_buffer,
        check_interval=args.check_interval,
        enable_ftmo_protection=args.ftmo_protection,
        ftmo_account_size=args.ftmo_account_size
    )


if __name__ == "__main__":
    main()