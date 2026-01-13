# scripts/live_monitor.py
"""
Live monitoring script - checks for Brooks signals every 5 minutes
Sends notifications via Telegram when signals are found
"""
import sys
import os
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import TrendParams, Trend, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from strategies.regime import RegimeParams, should_trade_today
from execution.guardrails import Guardrails, apply_guardrails
from execution.risk_manager import RiskManager
from utils.telegram_bot import TelegramBot, TradingSignal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


def is_ny_session_active() -> bool:
    """Check if we're currently in NY session (09:30-15:00 EST)"""
    now_ny = pd.Timestamp.now(tz=NY_TZ)
    current_time = now_ny.time()

    # NY session: 09:30 - 15:00
    session_start = pd.Timestamp("09:30").time()
    session_end = pd.Timestamp("15:00").time()

    return session_start <= current_time <= session_end


def check_for_signals(
        symbol: str,
        risk_pct: float,
        regime_filter: bool,
        chop_threshold: float,
        stop_buffer: float,
        telegram_bot: TelegramBot
) -> bool:
    """
    Check for trading signals and send Telegram notification if found

    Returns:
        True if signal found and sent
    """
    try:
        # Initialize MT5
        client = Mt5Client(mt5_module=mt5)
        if not client.initialize():
            logger.error("❌ Failed to connect to MT5")
            telegram_bot.send_error("MT5 connection failed")
            return False

        # Get symbol spec
        spec = client.get_symbol_specification(symbol)
        if spec is None:
            logger.error(f"❌ Symbol {symbol} not found")
            client.shutdown()
            return False

        # Fetch data
        logger.info(f"📊 Fetching data for {symbol}...")
        m15 = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, 300))
        m5 = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, 500))

        if m15.empty or m5.empty:
            logger.warning("⚠️ Empty data received")
            client.shutdown()
            return False

        # Check regime if enabled
        regime_status = "⚠️ NOT FILTERED"
        if regime_filter:
            regime_params = RegimeParams(chop_threshold=chop_threshold)
            should_trade, regime_reason = should_trade_today(m15, regime_params)

            if not should_trade:
                logger.info(f"⛔ {regime_reason}")
                client.shutdown()
                return False

            regime_status = "✅ TRENDING"
            logger.info(f"✅ {regime_reason}")

        # Infer trend
        trend_params = TrendParams(ema_period=20)
        trend_res, metrics = infer_trend_m15(m15, trend_params)

        logger.info(
            f"📈 Trend: {trend_res} (close={metrics.last_close:.2f}, "
            f"ema={metrics.last_ema:.2f}, slope={metrics.ema_slope:.4f})"
        )

        if trend_res not in [Trend.BULL, Trend.BEAR]:
            logger.info("↔️ No clear trend - no signal")
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
            m5,
            trend=side,
            spec=spec,
            p=strat_params,
            timeframe_minutes=5,
            now_utc=pd.Timestamp.now(tz="UTC"),
        )

        if planned_trade is None:
            logger.info("↔️ No signal found")
            client.shutdown()
            return False

        # Apply guardrails
        g = Guardrails(
            session_tz=NY_TZ,
            day_tz=NY_TZ,
            session_start="09:30",
            session_end="15:00",
            max_trades_per_day=2,
        )

        accepted, rejected = apply_guardrails([planned_trade], g)

        if not accepted:
            reason = rejected[0][1] if rejected else "Unknown"
            logger.info(f"⛔ Signal rejected by guardrails: {reason}")
            client.shutdown()
            return False

        trade = accepted[0]

        # Calculate position size
        acc_info = mt5.account_info()
        if acc_info is None:
            logger.error("❌ Could not fetch account info")
            client.shutdown()
            return False

        rm = RiskManager(risk_per_trade_pct=risk_pct)
        lots = rm.calculate_lot_size(
            balance=acc_info.balance,
            spec=spec,
            entry=trade.entry,
            stop=trade.stop
        )

        risk_usd = acc_info.balance * risk_pct / 100

        # Build signal
        signal = TradingSignal(
            symbol=symbol,
            side=trade.side.value,
            entry=trade.entry,
            stop=trade.stop,
            target=trade.tp,
            lots=lots,
            risk_usd=risk_usd,
            risk_pct=risk_pct,
            reason=trade.reason,
            regime=regime_status
        )

        # Send to Telegram
        logger.info(f"📤 Sending {trade.side.value} signal to Telegram...")
        success = telegram_bot.send_signal(signal)

        if success:
            logger.info("✅ Signal sent successfully!")

            # Also print to console
            print("\n" + "!" * 60)
            print(f" 🎯 SIGNAL FOUND & SENT TO TELEGRAM")
            print(f" Symbol    : {symbol}")
            print(f" Direction : {trade.side.value}")
            print(f" Entry     : {trade.entry:.2f}")
            print(f" Stop      : {trade.stop:.2f}")
            print(f" Target    : {trade.tp:.2f}")
            print(f" Lots      : {lots:.2f}")
            print(f" Risk      : ${risk_usd:.2f} ({risk_pct}%)")
            print(f" Regime    : {regime_status}")
            print("!" * 60 + "\n")
        else:
            logger.error("❌ Failed to send Telegram notification")

        client.shutdown()
        return success

    except Exception as e:
        logger.error(f"❌ Error checking signals: {e}")
        try:
            telegram_bot.send_error(f"Signal check error: {str(e)}")
        except:
            pass
        return False


def run_monitor(
        symbol: str,
        risk_pct: float,
        regime_filter: bool,
        chop_threshold: float,
        stop_buffer: float,
        check_interval: int = 300,  # 5 minutes
):
    """
    Main monitoring loop

    Args:
        symbol: Trading symbol
        risk_pct: Risk per trade (%)
        regime_filter: Enable regime filter
        chop_threshold: Chop threshold
        stop_buffer: Stop buffer
        check_interval: Seconds between checks (default 300 = 5 min)
    """
    logger.info("=" * 60)
    logger.info("🤖 BROOKS LIVE MONITOR STARTING")
    logger.info("=" * 60)
    logger.info(f"Symbol           : {symbol}")
    logger.info(f"Risk per trade   : {risk_pct}%")
    logger.info(f"Regime filter    : {'ENABLED' if regime_filter else 'DISABLED'}")
    if regime_filter:
        logger.info(f"Chop threshold   : {chop_threshold}")
    logger.info(f"Stop buffer      : {stop_buffer}")
    logger.info(f"Check interval   : {check_interval}s ({check_interval / 60:.1f} min)")
    logger.info(f"NY Session hours : 09:30-15:00 EST")
    logger.info("=" * 60 + "\n")

    # Initialize Telegram bot
    try:
        telegram_bot = TelegramBot()
        logger.info("✅ Telegram bot connected")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Telegram bot: {e}")
        logger.error("Check your .env file has TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return

    # Send startup notification
    telegram_bot._send_message(
        "🤖 <b>Brooks Live Monitor Started</b>\n\n"
        f"Symbol: {symbol}\n"
        f"Risk: {risk_pct}%\n"
        f"Regime Filter: {'ON' if regime_filter else 'OFF'}\n"
        f"Monitoring NY session (09:30-15:00 EST)"
    )

    iteration = 0
    last_signal_time = None

    try:
        while True:
            iteration += 1
            now = datetime.now()

            # Check if in NY session
            if not is_ny_session_active():
                logger.info(f"⏸️ Outside NY session - sleeping {check_interval}s...")
                time.sleep(check_interval)
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"🔍 CHECK #{iteration} - {now.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'=' * 60}")

            # Check for signals
            found = check_for_signals(
                symbol=symbol,
                risk_pct=risk_pct,
                regime_filter=regime_filter,
                chop_threshold=chop_threshold,
                stop_buffer=stop_buffer,
                telegram_bot=telegram_bot
            )

            if found:
                last_signal_time = now

            # Sleep until next check
            logger.info(f"💤 Sleeping {check_interval}s until next check...\n")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("\n⛔ Monitor stopped by user")
        telegram_bot._send_message("⛔ <b>Brooks Live Monitor Stopped</b>")
    except Exception as e:
        logger.error(f"\n❌ Monitor crashed: {e}")
        telegram_bot.send_error(f"Monitor crashed: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brooks Live Signal Monitor")
    parser.add_argument("--symbol", default="US500.cash", help="Trading symbol")
    parser.add_argument("--risk-pct", type=float, default=0.5,
                        help="Risk per trade (%%)")
    parser.add_argument("--regime-filter", action="store_true",
                        help="Enable regime filter")
    parser.add_argument("--chop-threshold", type=float, default=2.0,
                        help="Chop threshold")
    parser.add_argument("--stop-buffer", type=float, default=1.0,
                        help="Stop buffer")
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds (default: 300 = 5min)")

    args = parser.parse_args()

    run_monitor(
        symbol=args.symbol,
        risk_pct=args.risk_pct,
        regime_filter=args.regime_filter,
        chop_threshold=args.chop_threshold,
        stop_buffer=args.stop_buffer,
        check_interval=args.interval
    )