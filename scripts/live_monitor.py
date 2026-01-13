# scripts/live_monitor.py
"""
Brooks Live Trading Monitor
Checks for signals every 5 minutes during NY session
"""
import sys
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import MetaTrader5 as mt5

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import infer_trend_m15, TrendParams, Trend
from strategies.h2l2 import plan_next_open_trade, H2L2Params, Side
from strategies.regime import should_trade_today, RegimeParams
from execution.risk_manager import RiskManager
from execution.guardrails import Guardrails, apply_guardrails

# PRODUCTION CONFIG - EXACT zoals in backtest!
SYMBOL = "US500.cash"
RISK_PCT = 0.5  # Start conservatief!
MAX_TRADES_DAY = 2

# Strategy params (FROM BACKTEST)
TREND_PARAMS = TrendParams(
    ema_period=20,
    min_slope=0.15,
)

REGIME_PARAMS = RegimeParams(
    chop_threshold=2.0,  # EXACT zoals backtest
)

STRAT_PARAMS = H2L2Params(
    pullback_bars=3,
    signal_close_frac=0.30,
    min_risk_price_units=2.0,
    stop_buffer=1.0,  # EXACT zoals backtest!
    cooldown_bars=0,  # EXACT zoals backtest!
)


def check_for_signal():
    """
    Check if there's a trade signal RIGHT NOW

    Returns:
        dict with trade info, or None
    """
    # Connect MT5
    client = Mt5Client(mt5)
    if not client.initialize():
        print("❌ MT5 connection failed")
        return None

    try:
        # Fetch data
        m15_req = RatesRequest(SYMBOL, mt5.TIMEFRAME_M15, 300)
        m5_req = RatesRequest(SYMBOL, mt5.TIMEFRAME_M5, 500)

        m15 = fetch_rates(mt5, m15_req)
        m5 = fetch_rates(mt5, m5_req)

        if m15.empty or m5.empty:
            print("⚠️  Empty data")
            return None

        # 1. CHECK REGIME
        should_trade, reason = should_trade_today(m15, REGIME_PARAMS)
        if not should_trade:
            print(f"⛔ {datetime.now().strftime('%H:%M:%S')} - Market choppy: {reason}")
            return None

        # 2. CHECK TREND
        trend, metrics = infer_trend_m15(m15, TREND_PARAMS)
        if trend not in [Trend.BULL, Trend.BEAR]:
            print(f"⏸️  {datetime.now().strftime('%H:%M:%S')} - No clear trend (slope={metrics.ema_slope:.2f})")
            return None

        side = Side.LONG if trend == Trend.BULL else Side.SHORT

        # 3. CHECK FOR SETUP
        spec = client.get_symbol_specification(SYMBOL)
        if not spec:
            return None

        trade = plan_next_open_trade(
            m5,
            trend=side,
            spec=spec,
            p=STRAT_PARAMS,
            timeframe_minutes=5,
            now_utc=pd.Timestamp.now(tz="UTC"),
        )

        if not trade:
            return None

        # 4. CHECK GUARDRAILS (session time)
        g = Guardrails(
            session_tz="America/New_York",
            day_tz="America/New_York",
            session_start="09:30",
            session_end="15:00",
            max_trades_per_day=MAX_TRADES_DAY,
        )

        accepted, rejected = apply_guardrails([trade], g)
        if not accepted:
            reason = rejected[0][1] if rejected else "unknown"
            print(f"⏸️  Signal rejected: {reason}")
            return None

        # 5. CALCULATE POSITION SIZE
        acc = mt5.account_info()
        if not acc:
            return None

        rm = RiskManager(risk_per_trade_pct=RISK_PCT)
        lots = rm.calculate_lot_size(
            balance=acc.balance,
            spec=spec,
            entry=trade.entry,
            stop=trade.stop,
        )

        if lots <= 0:
            print("⚠️  Position size too small")
            return None

        # SUCCESS!
        return {
            "trade": trade,
            "lots": lots,
            "balance": acc.balance,
            "risk_usd": acc.balance * (RISK_PCT / 100),
            "risk_r": abs(trade.entry - trade.stop),
            "trend_metrics": metrics,
        }

    finally:
        client.shutdown()


def is_ny_trading_hours():
    """Check if it's currently NY trading hours"""
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)

    # Monday-Friday
    if now_ny.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # 09:30 - 15:00
    time_now = now_ny.time()
    start = datetime.strptime("09:30", "%H:%M").time()
    end = datetime.strptime("15:00", "%H:%M").time()

    return start <= time_now <= end


def main():
    """Main monitoring loop"""
    print("=" * 60)
    print("  🤖 BROOKS LIVE MONITOR")
    print("=" * 60)
    print(f"  Symbol       : {SYMBOL}")
    print(f"  Risk per trade: {RISK_PCT}%")
    print(f"  Max trades/day: {MAX_TRADES_DAY}")
    print("=" * 60)
    print(f"\n✅ Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⏰ Checking every 5 minutes during NY session (09:30-15:00)")
    print("Press Ctrl+C to stop\n")

    last_signal_time = None

    try:
        while True:
            now = datetime.now()

            # Only check during NY trading hours
            if is_ny_trading_hours():
                print(f"🔍 {now.strftime('%H:%M:%S')} - Checking for signal...")

                signal = check_for_signal()

                if signal:
                    # Prevent duplicate alerts (within 5 min)
                    if last_signal_time and (now - last_signal_time).seconds < 300:
                        print("   (Same signal as before, skipping alert)")
                    else:
                        print("\n" + "🚨" * 30)
                        print(f"  ⚡ SIGNAL DETECTED: {signal['trade'].side}")
                        print("=" * 60)
                        print(f"  Entry : {signal['trade'].entry:.2f}")
                        print(f"  Stop  : {signal['trade'].stop:.2f}")
                        print(f"  TP    : {signal['trade'].tp:.2f}")
                        print(f"  Lots  : {signal['lots']:.2f}")
                        print(f"  Risk  : ${signal['risk_usd']:.2f} ({signal['risk_r']:.1f} pts)")
                        print(f"  Reason: {signal['trade'].reason}")
                        print("=" * 60)
                        print("🚨" * 30 + "\n")

                        last_signal_time = now

                        # TODO: Send Telegram notification
                        # TODO: Auto-execute (if enabled)

                else:
                    print("   ✓ No signal")

            else:
                # Outside trading hours
                if now.hour == 9 and now.minute < 30:
                    print(f"⏰ {now.strftime('%H:%M:%S')} - Market opens at 09:30 NY time")

            # Sleep 5 minutes
            time.sleep(300)

    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
        print(f"Ran until {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()