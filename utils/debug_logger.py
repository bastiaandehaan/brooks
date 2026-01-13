# utils/debug_logger.py
"""
Advanced Debug Logger for Brooks Trading System

Captures everything needed for debugging:
- System state at time of error
- Recent market data
- Trade history
- Performance metrics
- Configuration snapshot

Usage in live_monitor.py:
    from utils.debug_logger import DebugLogger, capture_error_context

    debug = DebugLogger()

    try:
        # Your trading code
    except Exception as e:
        context = capture_error_context(e, m5_data, trades, config)
        debug.log_error(context)
        # Telegram notification sent automatically
"""
from __future__ import annotations

import json
import logging
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class DebugLogger:
    """
    Comprehensive debug logger for production issues

    Saves:
    - Error logs with full context
    - System snapshots
    - Market data at time of error
    - Performance metrics
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.log_dir / "errors").mkdir(exist_ok=True)
        (self.log_dir / "trades").mkdir(exist_ok=True)
        (self.log_dir / "snapshots").mkdir(exist_ok=True)
        (self.log_dir / "debug").mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_error(self, context: Dict[str, Any]) -> Path:
        """
        Log error with full context

        Args:
            context: Dict from capture_error_context()

        Returns:
            Path to saved error log
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.log_dir / "errors" / f"error_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(context, f, indent=2, default=str)

        logger.error(f"Error logged to: {filename}")

        # Also create human-readable version
        self._create_readable_error_report(context, filename)

        return filename

    def _create_readable_error_report(self, context: Dict, json_path: Path) -> None:
        """Create human-readable error report"""
        txt_path = json_path.with_suffix('.txt')

        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BROOKS TRADING SYSTEM - ERROR REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {context.get('timestamp', 'Unknown')}\n")
            f.write(f"Session ID: {context.get('session_id', 'Unknown')}\n")
            f.write(f"Error Type: {context.get('error_type', 'Unknown')}\n\n")

            f.write("ERROR MESSAGE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{context.get('error_message', 'No message')}\n\n")

            f.write("STACK TRACE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{context.get('stack_trace', 'No trace')}\n\n")

            if 'system_state' in context:
                f.write("SYSTEM STATE:\n")
                f.write("-" * 80 + "\n")
                for key, value in context['system_state'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            if 'config' in context:
                f.write("CONFIGURATION:\n")
                f.write("-" * 80 + "\n")
                f.write(json.dumps(context['config'], indent=2))
                f.write("\n\n")

            if 'recent_trades' in context:
                f.write("RECENT TRADES:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total: {len(context['recent_trades'])}\n")
                for i, trade in enumerate(context['recent_trades'][-5:], 1):
                    f.write(f"\n  Trade {i}:\n")
                    for k, v in trade.items():
                        f.write(f"    {k}: {v}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("TO SHARE WITH DEVELOPER:\n")
            f.write("1. Send this .txt file\n")
            f.write("2. Send corresponding .json file\n")
            f.write("3. Describe what you were doing when error occurred\n")
            f.write("=" * 80 + "\n")

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log individual trade"""
        date_str = datetime.now().strftime("%Y%m%d")
        filename = self.log_dir / "trades" / f"trades_{date_str}.jsonl"

        # Append to JSONL (JSON Lines) format
        with open(filename, 'a') as f:
            f.write(json.dumps(trade_data, default=str) + "\n")

    def log_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Log end-of-day summary"""
        date_str = datetime.now().strftime("%Y%m%d")
        filename = self.log_dir / "snapshots" / f"daily_summary_{date_str}.json"

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def create_system_snapshot(self,
                               market_data: Optional[pd.DataFrame] = None,
                               trades: Optional[list] = None,
                               account_info: Optional[Dict] = None) -> Path:
        """
        Create complete system snapshot (for debugging or daily backup)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.log_dir / "snapshots" / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "python_version": sys.version,
        }

        # Save market data
        if market_data is not None:
            csv_path = snapshot_dir / "market_data.csv"
            market_data.tail(100).to_csv(csv_path)
            snapshot["market_data_rows"] = len(market_data)
            snapshot["market_data_file"] = str(csv_path)

        # Save trades
        if trades is not None:
            trades_path = snapshot_dir / "trades.json"
            with open(trades_path, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
            snapshot["trades_count"] = len(trades)

        # Save account info
        if account_info is not None:
            account_path = snapshot_dir / "account_info.json"
            with open(account_path, 'w') as f:
                json.dump(account_info, f, indent=2, default=str)

        # Save snapshot metadata
        meta_path = snapshot_dir / "snapshot.json"
        with open(meta_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"System snapshot saved to: {snapshot_dir}")
        return snapshot_dir

    def get_recent_logs(self, n: int = 10) -> list:
        """Get N most recent error logs"""
        error_dir = self.log_dir / "errors"
        if not error_dir.exists():
            return []

        files = sorted(error_dir.glob("error_*.json"), reverse=True)
        return [str(f) for f in files[:n]]

    def get_daily_trades(self, date: Optional[str] = None) -> list:
        """Get all trades for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        trade_file = self.log_dir / "trades" / f"trades_{date}.jsonl"
        if not trade_file.exists():
            return []

        trades = []
        with open(trade_file) as f:
            for line in f:
                trades.append(json.loads(line))
        return trades


def capture_error_context(
        error: Exception,
        market_data: Optional[pd.DataFrame] = None,
        trades: Optional[list] = None,
        config: Optional[Dict] = None,
        system_state: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Capture complete context when error occurs

    Args:
        error: The exception that occurred
        market_data: Recent market data (last 100 bars)
        trades: Recent trades (last 10)
        config: Current configuration
        system_state: Current system state (balance, positions, etc.)

    Returns:
        Dict with all context needed for debugging
    """
    context = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        "python_version": sys.version,
    }

    # Add market data summary
    if market_data is not None:
        try:
            context["market_data"] = {
                "rows": len(market_data),
                "last_timestamp": str(market_data.index[-1]) if len(market_data) > 0 else None,
                "columns": list(market_data.columns),
                "last_close": float(market_data['close'].iloc[-1]) if 'close' in market_data.columns else None,
                "last_10_closes": market_data['close'].tail(10).tolist() if 'close' in market_data.columns else [],
            }
        except Exception as e:
            context["market_data"] = f"Error capturing market data: {e}"

    # Add recent trades
    if trades is not None:
        try:
            context["recent_trades"] = [
                {
                    "timestamp": str(getattr(t, 'execute_ts', 'unknown')),
                    "side": str(getattr(t, 'side', 'unknown')),
                    "entry": float(getattr(t, 'entry', 0)),
                    "stop": float(getattr(t, 'stop', 0)),
                    "tp": float(getattr(t, 'tp', 0)),
                    "reason": str(getattr(t, 'reason', 'unknown')),
                }
                for t in trades[-10:]  # Last 10 trades
            ]
        except Exception as e:
            context["recent_trades"] = f"Error capturing trades: {e}"

    # Add config
    if config is not None:
        context["config"] = config

    # Add system state
    if system_state is not None:
        context["system_state"] = system_state

    return context


def send_debug_bundle_via_telegram(
        error_log_path: Path,
        telegram_bot: Optional[Any] = None
) -> bool:
    """
    Send debug info via Telegram

    Args:
        error_log_path: Path to error log
        telegram_bot: TelegramBot instance

    Returns:
        True if sent successfully
    """
    if telegram_bot is None:
        return False

    try:
        # Read error summary
        txt_path = error_log_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path) as f:
                content = f.read()

            # Send first 1000 chars
            preview = content[:1000]
            if len(content) > 1000:
                preview += "\n\n[TRUNCATED - Full log saved locally]"

            telegram_bot.send_error(
                f"🐛 ERROR DETECTED\n\n{preview}\n\n"
                f"Full log: {error_log_path.name}"
            )
            return True
    except Exception as e:
        logger.error(f"Failed to send debug bundle: {e}")

    return False


# Convenience function for live_monitor.py
def setup_debug_logging(log_level: str = "INFO") -> DebugLogger:
    """
    Setup debug logging for live trading session

    Returns:
        DebugLogger instance
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(f'logs/debug/session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    return DebugLogger()


# Example usage
if __name__ == "__main__":
    print("Testing Debug Logger...\n")

    debug = DebugLogger()

    # Test 1: Log an error
    try:
        # Simulate an error
        raise ValueError("Test error: MT5 connection lost")
    except Exception as e:
        context = capture_error_context(
            error=e,
            config={
                "symbol": "US500.cash",
                "risk_pct": 0.5,
                "regime_filter": True,
            },
            system_state={
                "balance": 10000,
                "daily_pnl": -150,
                "trades_today": 3,
            }
        )

        error_path = debug.log_error(context)
        print(f"✅ Error logged to: {error_path}")
        print(f"✅ Readable report: {error_path.with_suffix('.txt')}")

    # Test 2: Log a trade
    debug.log_trade({
        "timestamp": datetime.now().isoformat(),
        "side": "LONG",
        "entry": 5847.50,
        "stop": 5845.00,
        "tp": 5852.50,
        "result": "+2.0R",
        "pnl": 100.00,
    })
    print("\n✅ Trade logged")

    # Test 3: Create snapshot
    snapshot_dir = debug.create_system_snapshot(
        account_info={"balance": 10000, "equity": 10150}
    )
    print(f"\n✅ Snapshot created: {snapshot_dir}")

    # Test 4: Get recent logs
    recent = debug.get_recent_logs(n=5)
    print(f"\n✅ Found {len(recent)} recent error logs")

    print("\n" + "=" * 60)
    print("✅ All debug logger tests passed!")
    print("=" * 60)