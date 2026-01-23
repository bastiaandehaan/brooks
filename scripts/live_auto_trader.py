#!/usr/bin/env python3
"""
Brooks Auto Trader - AUTOMATIC ORDER PLACEMENT
⚠️ USE WITH EXTREME CAUTION - ONLY ON DEMO FIRST!
"""

import argparse
import logging
import os
import sys
import time

import MetaTrader5 as mt5
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.ftmo_guardian import FTMOGuardian, FTMORules
from execution.ftmo_state import FTMOState
from execution.guardrails import Guardrails
from execution.risk_manager import RiskManager
from execution.trade_executor import execute_trade_ftmo_safe
from strategies.config import StrategyConfig
from strategies.context import Trend, infer_trend_m15
from strategies.h2l2 import Side, plan_next_open_trade
from strategies.regime import should_trade_today
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates

logger = logging.getLogger(__name__)


def auto_trade_loop(
    strategy_config: StrategyConfig,
    ftmo_config: dict,
    check_interval: int = 60,
):
    """
    Main auto-trading loop - PLACES REAL ORDERS!
    """
    # Initialize MT5
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        logger.error("❌ MT5 init failed")
        return

    # Get initial equity
    acc_info = mt5.account_info()
    if not acc_info:
        logger.error("❌ Failed to get account info")
        client.shutdown()
        return

    initial_equity = float(acc_info.equity)
    running_max_equity = initial_equity

    # CRITICAL: Verify DEMO account
    acc_server = acc_info.server.lower()
    is_demo = "demo" in acc_server

    if not is_demo:
        logger.error("🚨 LIVE ACCOUNT DETECTED - ABORTING FOR SAFETY!")
        logger.error(f"   Server: {acc_info.server}")
        logger.error("   This script should ONLY run on demo accounts!")
        client.shutdown()
        return

    logger.info("✅ DEMO account verified: %s", acc_info.server)

    # Initialize FTMO protection
    if ftmo_config.get("enabled", False):
        account_size = ftmo_config.get("account_size", 10000)
        rules = FTMORules(
            account_type=FTMORules.for_10k_challenge().account_type,
            initial_balance=account_size,
            max_daily_loss_pct=ftmo_config.get("max_daily_loss_pct", 5.0),
            max_total_loss_pct=ftmo_config.get("max_total_loss_pct", 10.0),
            profit_target_pct=ftmo_config.get("profit_target_pct", 10.0),
            daily_loss_buffer_pct=ftmo_config.get("daily_buffer_pct", 1.0),
            total_loss_buffer_pct=ftmo_config.get("total_buffer_pct", 2.0),
        )
        ftmo_guardian = FTMOGuardian(rules=rules)
        ftmo_state = FTMOState.initialize(initial_equity, day_tz="America/New_York")
        logger.info("✅ FTMO Protection: ENABLED")
    else:
        ftmo_guardian = None
        ftmo_state = None
        logger.warning("⚠️ FTMO Protection: DISABLED - HIGH RISK!")

    # Initialize components
    risk_manager = RiskManager()
    spec = client.get_symbol_specification(strategy_config.symbol)

    if not spec:
        logger.error("❌ Failed to get symbol spec for %s", strategy_config.symbol)
        client.shutdown()
        return

    logger.info("=" * 80)
    logger.info("🤖 AUTO TRADER STARTED - ORDERS WILL BE PLACED AUTOMATICALLY")
    logger.info("=" * 80)
    logger.info(f"Account: {acc_info.login} ({acc_info.server})")
    logger.info(f"Balance: ${acc_info.balance:,.2f}")
    logger.info(f"Symbol: {strategy_config.symbol}")
    logger.info(f"Risk per trade: {strategy_config.risk_pct}%")
    logger.info(f"Max trades/day: {strategy_config.guardrails.max_trades_per_day}")
    logger.info(f"Check interval: {check_interval}s")
    logger.info("=" * 80)
    logger.warning("⚠️ AUTOMATIC ORDER PLACEMENT IS ACTIVE!")
    logger.info("=" * 80)

    iteration = 0
    last_ftmo_log = time.time()

    try:
        while True:
            iteration += 1

            # Emergency stop check
            if os.path.exists("STOP.txt"):
                logger.error("🛑 EMERGENCY STOP DETECTED - HALTING")
                break

            # Update equity
            acc_info = mt5.account_info()
            if acc_info:
                equity_now = float(acc_info.equity)
                running_max_equity = max(running_max_equity, equity_now)
            else:
                equity_now = running_max_equity
                logger.warning("⚠️ Could not fetch account info, using cached equity")

            # Update FTMO state
            if ftmo_state:
                now_utc = pd.Timestamp.now(tz="UTC")
                day_reset = ftmo_state.update(equity_now, now_utc)
                if day_reset:
                    logger.info("🔄 New trading day - limits reset")

                # Log FTMO status every 5 minutes
                if time.time() - last_ftmo_log >= 300:
                    daily_pnl = ftmo_state.get_daily_pnl(equity_now)
                    total_pnl = ftmo_state.get_total_pnl(equity_now)

                    logger.info("=" * 80)
                    logger.info("💼 FTMO STATUS")
                    logger.info("=" * 80)
                    logger.info(f"Equity: ${equity_now:,.2f}")
                    logger.info(f"Daily P&L: ${daily_pnl:+,.2f}")
                    logger.info(f"Total P&L: ${total_pnl:+,.2f}")
                    logger.info(f"Trading Days: {ftmo_state.trading_days}")

                    if ftmo_guardian:
                        max_risk = ftmo_guardian.get_max_allowed_risk(equity_now, daily_pnl)
                        logger.info(f"Max Risk Headroom: ${max_risk:,.2f}")

                    logger.info("=" * 80)
                    last_ftmo_log = time.time()

            logger.info(f"[Iter {iteration}] Checking for signals...")

            # Fetch data
            try:
                req_m15 = RatesRequest(strategy_config.symbol, mt5.TIMEFRAME_M15, 300)
                req_m5 = RatesRequest(strategy_config.symbol, mt5.TIMEFRAME_M5, 500)

                m15_data = fetch_rates(mt5, req_m15)
                m5_data = fetch_rates(mt5, req_m5)
            except Exception as e:
                logger.error(f"❌ Data fetch failed: {e}")
                time.sleep(check_interval)
                continue

            if m15_data.empty or m5_data.empty:
                logger.warning("⚠️ Empty data - skipping")
                time.sleep(check_interval)
                continue

            # Regime filter
            if strategy_config.regime_filter:
                ok, reason = should_trade_today(m15_data, strategy_config.regime_params)
                if not ok:
                    logger.info(f"⛔ Regime filter: {reason}")
                    time.sleep(check_interval)
                    continue
                logger.info(f"✅ Regime: {reason}")

            # Trend detection
            trend, metrics = infer_trend_m15(m15_data, strategy_config.trend_params)
            logger.info(f"📊 Trend: {trend.value if hasattr(trend, 'value') else str(trend)}")

            if trend not in (Trend.BULL, Trend.BEAR):
                logger.info("⏸️ No clear trend - waiting")
                time.sleep(check_interval)
                continue

            side = Side.LONG if trend == Trend.BULL else Side.SHORT

            # Plan trade
            planned = plan_next_open_trade(
                m5_data,
                side,
                spec,
                strategy_config.h2l2_params,
                timeframe_minutes=5,
            )

            if not planned:
                logger.info("⏸️ No setup found")
                time.sleep(check_interval)
                continue

            logger.info("=" * 80)
            logger.info("🎯 SETUP DETECTED!")
            logger.info("=" * 80)
            logger.info(f"Reason: {planned.reason}")
            logger.info(f"Side: {planned.side.value}")
            logger.info(f"Entry: {planned.entry:.2f}")
            logger.info(f"Stop: {planned.stop:.2f}")
            logger.info(f"Target: {planned.tp:.2f}")
            logger.info("=" * 80)

            # ===================================================
            # 🚨 EXECUTE TRADE AUTOMATICALLY (ATOMIC + FTMO-SAFE)
            # ===================================================
            logger.info("🚀 Executing trade...")

            result = execute_trade_ftmo_safe(
                planned_trade=planned,
                equity_now=equity_now,
                ftmo_state=ftmo_state,
                ftmo_guardian=ftmo_guardian,
                guardrails=strategy_config.guardrails,
                risk_manager=risk_manager,
                spec=spec,
                requested_risk_pct=strategy_config.risk_pct,
                mt5_module=mt5,
            )

            if result.success:
                logger.info("=" * 80)
                logger.info("✅ ✅ ✅ TRADE EXECUTED SUCCESSFULLY ✅ ✅ ✅")
                logger.info("=" * 80)
                logger.info(f"Ticket: {result.ticket}")
                logger.info(f"Side: {planned.side.value}")
                logger.info(f"Entry: {result.filled_price:.2f}")
                logger.info(f"Stop: {planned.stop:.2f}")
                logger.info(f"Target: {planned.tp:.2f}")
                logger.info(f"Lots: {result.filled_lots:.2f}")
                logger.info(f"Risk: ${result.actual_risk_usd:.2f}")
                logger.info("=" * 80)

                # Increment trading days counter
                if ftmo_state:
                    ftmo_state.increment_trading_day()

            else:
                logger.warning("=" * 80)
                logger.warning("⛔ TRADE BLOCKED")
                logger.warning("=" * 80)
                logger.warning(f"Reason: {result.reason}")
                if result.block_stage:
                    logger.warning(f"Stage: {result.block_stage}")
                logger.warning("=" * 80)

            # Wait before next check
            logger.info(f"💤 Sleeping {check_interval}s until next check...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("⛔ Auto trader stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Auto trader crashed: {e}", exc_info=True)
    finally:
        logger.info("🔌 Shutting down MT5 connection...")
        client.shutdown()
        logger.info("✅ Auto trader stopped cleanly")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Brooks Auto Trader - AUTOMATIC ORDER PLACEMENT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This script places REAL orders automatically!

    ONLY use on DEMO accounts initially!

    Emergency stop: Create STOP.txt in project root
""",
    )
    parser.add_argument("--strategy", default="config/strategies/us500_sniper.yaml")
    parser.add_argument("--env", default="config/environments/production.yaml")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"logs/auto_trader_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
        ],
    )

    # Load configs
    logger.info("📋 Loading configuration...")
    strategy_config = StrategyConfig.load(args.strategy)

    with open(args.env, "r") as f:
        env_config = yaml.safe_load(f)
    ftmo_config = env_config.get("ftmo", {})

    # Display configuration
    print("\n" + "=" * 80)
    print("  📋 CONFIGURATION LOADED")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {strategy_config.symbol}")
    print(f"Risk per trade: {strategy_config.risk_pct}%")
    print(f"Max trades/day: {strategy_config.guardrails.max_trades_per_day}")
    print(f"Regime filter: {'ON' if strategy_config.regime_filter else 'OFF'}")
    print(f"FTMO Protection: {'ENABLED' if ftmo_config.get('enabled') else 'DISABLED'}")
    print("=" * 80)

    # WARNING prompt
    print("\n" + "⚠️" * 40)
    print("  ⚠️  AUTOMATIC ORDER PLACEMENT IS ABOUT TO START  ⚠️")
    print("⚠️" * 40)
    print("\n⚠️ ORDERS WILL BE PLACED AUTOMATICALLY WITHOUT CONFIRMATION!")
    print("⚠️ Ensure you are on a DEMO account!")
    print("\n🛑 To stop trading, either:")
    print("   1. Press Ctrl+C")
    print("   2. Create STOP.txt in project root")
    print("\n⏰ Starting in 10 seconds...")
    print("   Press Ctrl+C NOW to abort!\n")

    try:
        for i in range(10, 0, -1):
            print(f"   {i}...", end="", flush=True)
            time.sleep(1)
        print(" GO!")
    except KeyboardInterrupt:
        print("\n\n✅ Aborted by user")
        return 0

    print("\n🚀 Starting auto trader...\n")

    auto_trade_loop(strategy_config, ftmo_config, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
