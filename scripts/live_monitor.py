#!/usr/bin/env python3
"""
Brooks Live Monitor - WITH FTMO PROTECTION

Key Changes for FTMO Integration:
1. Initialize FTMOState at startup
2. Update state every iteration (equity + timestamp)
3. Call trade_gate BEFORE sizing
4. Use capped risk for position sizing
5. Log FTMO status periodically

CRITICAL: All risk calculations use EQUITY (not balance)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.ftmo_guardian import FTMOGuardian

# FTMO imports
from execution.ftmo_state import FTMOState
from execution.guardrails import Guardrails, apply_guardrails
from execution.risk_manager import RiskManager
from execution.trade_gate import (
    check_ftmo_trade_gate,
    convert_risk_to_pct,
    log_ftmo_status,
)
from strategies.config import StrategyConfig
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from strategies.regime import RegimeParams, should_trade_today
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates

logger = logging.getLogger(__name__)
NY_TZ = "America/New_York"


def load_ftmo_config(env_config_path: str) -> dict:
    """Load FTMO configuration from environment YAML"""
    try:
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
        return env_config.get("ftmo", {})
    except Exception as e:
        logger.error("Failed to load FTMO config: %s", e)
        return {}


def initialize_ftmo_protection(
    ftmo_config: dict,
    initial_equity: float,
    day_tz: str = NY_TZ,
) -> Tuple[Optional[FTMOGuardian], Optional[FTMOState]]:
    """
    Initialize FTMO protection system.

    Returns:
        (guardian, state) or (None, None) if FTMO disabled
    """
    if not ftmo_config.get("enabled", False):
        logger.info("FTMO protection: DISABLED")
        return None, None

    logger.info("=" * 70)
    logger.info("🛡️  INITIALIZING FTMO PROTECTION")
    logger.info("=" * 70)

    # Import FTMORules
    from execution.ftmo_guardian import FTMOAccountType, FTMORules

    # Map account size to account type
    account_size = ftmo_config["account_size"]
    if account_size == 10000:
        rules = FTMORules.for_10k_challenge()
    elif account_size == 25000:
        rules = FTMORules(
            account_type=FTMOAccountType.CHALLENGE_25K,
            initial_balance=25000.0,
            profit_target_pct=10.0,
        )
    elif account_size == 50000:
        rules = FTMORules(
            account_type=FTMOAccountType.CHALLENGE_50K,
            initial_balance=50000.0,
            profit_target_pct=10.0,
        )
    else:
        # Custom size
        rules = FTMORules(
            account_type=FTMOAccountType.CHALLENGE_10K,
            initial_balance=account_size,
            profit_target_pct=ftmo_config.get("profit_target_pct", 10.0),
        )

    # Override buffers if specified in config
    if "daily_buffer_pct" in ftmo_config:
        rules.daily_loss_buffer_pct = ftmo_config["daily_buffer_pct"]
    if "total_buffer_pct" in ftmo_config:
        rules.total_loss_buffer_pct = ftmo_config["total_buffer_pct"]

    # Create guardian
    guardian = FTMOGuardian(rules=rules)

    # Create state tracker
    state = FTMOState.initialize(initial_equity, day_tz)

    logger.info("  Account Size: $%,.2f", rules.initial_balance)
    logger.info(
        "  Profit Target: $%,.2f (%.0f%%)",
        rules.initial_balance * rules.profit_target_pct / 100,
        rules.profit_target_pct,
    )
    logger.info("  Max Daily Loss: $%,.2f", guardian.max_daily_loss_usd)
    logger.info("  Max Total Loss: $%,.2f", guardian.max_total_loss_usd)
    logger.info("  Safe Daily (buffer): $%,.2f", guardian.safe_daily_loss_usd)
    logger.info("  Safe Total (buffer): $%,.2f", guardian.safe_total_loss_usd)
    logger.info("  Min Trading Days: %d", rules.min_trading_days)
    logger.info("  Initial Equity: $%,.2f", initial_equity)
    logger.info("=" * 70)

    return guardian, state


def check_for_signals_with_ftmo(
    *,
    symbol: str,
    strategy_config: StrategyConfig,
    ftmo_guardian: Optional[FTMOGuardian],
    ftmo_state: Optional[FTMOState],
    running_max_equity: float,
) -> bool:
    """
    Check for trading signals WITH FTMO protection.

    Returns:
        True if signal found and trade allowed, False otherwise
    """
    logger.info("🔍 Checking for signals...")

    try:
        # Connect to MT5
        client = Mt5Client(mt5_module=mt5)
        if not client.initialize():
            logger.error("❌ Failed to connect to MT5")
            return False

        # Get current account state
        acc_info = mt5.account_info()
        if not acc_info:
            logger.error("❌ Failed to get account info")
            client.shutdown()
            return False

        equity_now = float(acc_info.equity)
        balance_now = float(acc_info.balance)

        logger.info("💰 Account: equity=$%.2f, balance=$%.2f", equity_now, balance_now)

        # Update FTMO state (if enabled)
        if ftmo_state:
            now_utc = pd.Timestamp.now(tz="UTC")
            day_reset = ftmo_state.update(equity_now, now_utc)

            if day_reset:
                logger.info("🔄 New NY trading day - daily limits reset")

        # Check FTMO status (if enabled)
        if ftmo_guardian and ftmo_state:
            daily_pnl = ftmo_state.get_daily_pnl(equity_now)
            guardian_status = ftmo_guardian.get_status(equity_now, daily_pnl)

            if guardian_status["status"] in ["STOP_DAILY", "STOP_TOTAL"]:
                logger.error("⛔ FTMO BLOCKED: %s", guardian_status["status"])
                client.shutdown()
                return False

        # Get symbol spec
        spec = client.get_symbol_specification(symbol)
        if not spec:
            logger.error("❌ Symbol %s not found", symbol)
            client.shutdown()
            return False

        # Fetch data
        req_m15 = RatesRequest(symbol, mt5.TIMEFRAME_M15, 300)
        req_m5 = RatesRequest(symbol, mt5.TIMEFRAME_M5, 500)

        m15_data = fetch_rates(mt5, req_m15)
        m5_data = fetch_rates(mt5, req_m5)

        if m15_data.empty or m5_data.empty:
            logger.warning("⚠️ Empty dataframes")
            client.shutdown()
            return False

        # Regime filter (if enabled)
        if strategy_config.regime_filter:
            ok, reason = should_trade_today(m15_data, strategy_config.regime_params)
            if not ok:
                logger.info("⛔ Regime filter: %s", reason)
                client.shutdown()
                return False
            logger.info("✅ Regime: %s", reason)

        # Trend inference
        trend, _ = infer_trend_m15(m15_data, strategy_config.trend_params)
        logger.info("📊 Trend: %s", trend.value if hasattr(trend, "value") else str(trend))

        if trend not in (Trend.BULL, Trend.BEAR):
            logger.info("No clear trend")
            client.shutdown()
            return False

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
            logger.info("No setup found")
            client.shutdown()
            return False

        logger.info("✅ Setup found: %s", planned.reason)

        # === FTMO TRADE GATE (CRITICAL) ===
        # Calculate requested risk in USD
        requested_risk_usd = (strategy_config.risk_pct / 100.0) * equity_now

        if ftmo_guardian and ftmo_state:
            # Check FTMO gate
            gate_result = check_ftmo_trade_gate(
                equity_now=equity_now,
                ftmo_state=ftmo_state,
                ftmo_guardian=ftmo_guardian,
                requested_risk_usd=requested_risk_usd,
                min_risk_threshold=10.0,
            )

            if gate_result.blocked:
                logger.warning("⛔ FTMO GATE BLOCKED: %s", gate_result.reason)
                client.shutdown()
                return False

            # Use FTMO-capped risk
            risk_usd_final = gate_result.capped_risk_usd
            risk_pct_final = convert_risk_to_pct(risk_usd_final, equity_now)

            logger.info(
                "💰 Risk: requested=$%.2f, FTMO-capped=$%.2f (%.3f%%)",
                requested_risk_usd,
                risk_usd_final,
                risk_pct_final,
            )
        else:
            # No FTMO protection - use requested risk
            risk_usd_final = requested_risk_usd
            risk_pct_final = strategy_config.risk_pct

            logger.info(
                "💰 Risk: $%.2f (%.2f%%) - NO FTMO PROTECTION", risk_usd_final, risk_pct_final
            )

        # === POSITION SIZING ===
        rm = RiskManager()
        lots, final_risk_usd = rm.size_position(
            balance=equity_now,  # Use equity!
            risk_pct=risk_pct_final,
            entry=planned.entry,
            stop=planned.stop,
            tick_size=float(spec.tick_size),
            tick_value=float(spec.tick_value),
            contract_size=float(spec.contract_size),
        )

        if lots <= 0:
            logger.warning("⚠️ Position sizing rejected (lots=%s)", lots)
            client.shutdown()
            return False

        # === GUARDRAILS ===
        g = Guardrails(
            max_trades_per_day=strategy_config.guardrails.max_trades_per_day,
            session_start=strategy_config.guardrails.session_start,
            session_end=strategy_config.guardrails.session_end,
            day_tz=strategy_config.guardrails.day_tz,
            session_tz=strategy_config.guardrails.session_tz,
        )

        accepted, rejected = apply_guardrails([planned], g)
        if not accepted:
            logger.info("⛔ Guardrails rejected trade")
            client.shutdown()
            return False

        # === SIGNAL OUTPUT ===
        print("\n" + "!" * 70)
        print(f"  🎯 BROOKS TRADE SIGNAL: {symbol}")
        print("!" * 70)
        print(f"  Side: {planned.side.value}")
        print(f"  Entry: {planned.entry:.2f}")
        print(f"  Stop: {planned.stop:.2f}")
        print(f"  Target: {planned.tp:.2f}")
        print(f"  Volume: {lots:.2f} lots")
        print(f"  Risk: ${final_risk_usd:.2f} ({risk_pct_final:.2f}%)")
        print(f"  Reason: {planned.reason}")
        if ftmo_guardian:
            print(f"  FTMO Protected: ✅ (capped from ${requested_risk_usd:.2f})")
        print("!" * 70 + "\n")

        logger.info("📱 Signal ready - MANUAL EXECUTION REQUIRED")

        client.shutdown()
        return True

    except Exception as e:
        logger.error("❌ Error checking signals: %s", e, exc_info=True)
        return False


def run_monitor(
    *,
    strategy_config_path: str,
    env_config_path: str,
    check_interval: int,
) -> None:
    """
    Run live monitor with FTMO protection.
    """
    # Load configs
    strategy_config = StrategyConfig.load(strategy_config_path)
    ftmo_config = load_ftmo_config(env_config_path)

    # Initialize MT5 to get initial equity
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        logger.error("Failed to initialize MT5")
        return

    acc_info = mt5.account_info()
    if not acc_info:
        logger.error("Failed to get account info")
        client.shutdown()
        return

    initial_equity = float(acc_info.equity)
    client.shutdown()

    # Initialize FTMO protection
    ftmo_guardian, ftmo_state = initialize_ftmo_protection(
        ftmo_config,
        initial_equity,
        day_tz=NY_TZ,
    )

    # Track running max equity
    running_max_equity = initial_equity
    last_ftmo_log = time.time()
    ftmo_log_interval = ftmo_config.get("logging", {}).get("ftmo_status_interval", 300)

    logger.info("🤖 Brooks Live Monitor Started")
    logger.info("Strategy: %s", strategy_config_path)
    logger.info("Check interval: %ds", check_interval)

    iteration = 0
    try:
        while True:
            iteration += 1

            # Check emergency stop
            if os.path.exists("STOP.txt"):
                logger.warning("🛑 Emergency stop detected")
                break

            # Get current equity
            client = Mt5Client(mt5_module=mt5)
            if client.initialize():
                acc_info = mt5.account_info()
                if acc_info:
                    equity_now = float(acc_info.equity)
                    running_max_equity = max(running_max_equity, equity_now)
                client.shutdown()
            else:
                equity_now = running_max_equity

            # Log FTMO status periodically
            if ftmo_guardian and ftmo_state:
                if time.time() - last_ftmo_log >= ftmo_log_interval:
                    log_ftmo_status(equity_now, ftmo_state, ftmo_guardian, running_max_equity)
                    last_ftmo_log = time.time()

            logger.info("Iteration %d - checking signals...", iteration)

            # Check for signals
            check_for_signals_with_ftmo(
                symbol=strategy_config.symbol,
                strategy_config=strategy_config,
                ftmo_guardian=ftmo_guardian,
                ftmo_state=ftmo_state,
                running_max_equity=running_max_equity,
            )

            logger.info("💤 Sleeping %ds...", check_interval)
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("⛔ Monitor stopped by user")
    except Exception as e:
        logger.error("❌ Monitor crashed: %s", e, exc_info=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Brooks Live Monitor with FTMO Protection")
    parser.add_argument("--strategy", default="config/strategies/us500_sniper.yaml")
    parser.add_argument("--env", default="config/environments/production.yaml")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_monitor(
        strategy_config_path=args.strategy,
        env_config_path=args.env,
        check_interval=args.interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
