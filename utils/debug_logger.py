# utils/debug_logger.py
"""
Advanced Debug Logger for Brooks Trading System

Captures everything needed for debugging:
- System state at time of error
- Recent market data
- Trade history
- Performance metrics
- Configuration snapshot
"""
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


class DebugLogger:
    """Logs errors and system state for debugging"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.errors_dir = self.log_dir / "errors"
        self.trades_dir = self.log_dir / "trades"
        self.snapshots_dir = self.log_dir / "snapshots"
        self.debug_dir = self.log_dir / "debug"

        # Create directories
        for d in [self.errors_dir, self.trades_dir, self.snapshots_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def log_error(self, error_context: Dict[str, Any]) -> None:
        """Log error with full context"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON (machine readable)
        json_file = self.errors_dir / f"error_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(error_context, f, indent=2, default=str)

        # Save TXT (human readable)
        txt_file = self.errors_dir / f"error_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BROOKS TRADING SYSTEM - ERROR REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {error_context.get('timestamp', 'N/A')}\n")
            f.write(f"Error Type: {error_context.get('error_type', 'N/A')}\n")
            f.write(f"Error Message: {error_context.get('error_message', 'N/A')}\n\n")
            f.write("STACK TRACE:\n")
            f.write("-" * 80 + "\n")
            f.write(error_context.get('stack_trace', 'N/A'))
            f.write("\n\n")
            f.write("SYSTEM STATE:\n")
            f.write("-" * 80 + "\n")
            f.write(json.dumps(error_context.get('system_state', {}), indent=2, default=str))
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("TO SHARE WITH DEVELOPER:\n")
            f.write("1. Send this .txt file\n")
            f.write("2. Send corresponding .json file\n")
            f.write("3. Describe what you were doing\n")
            f.write("=" * 80 + "\n")

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log a trade to daily log file"""
        today = datetime.now().strftime("%Y%m%d")
        log_file = self.trades_dir / f"trades_{today}.jsonl"

        # Append trade (JSONL format - one JSON per line)
        with open(log_file, 'a') as f:
            f.write(json.dumps(trade_data, default=str) + "\n")

    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        """Save system snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.snapshots_dir / f"snapshot_{timestamp}.json"

        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2, default=str)

    def save_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Save end-of-day summary"""
        today = datetime.now().strftime("%Y%m%d")
        summary_file = self.snapshots_dir / f"daily_summary_{today}.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def capture_error_context(
        exception: Exception,
        market_data: Optional[pd.DataFrame] = None,
        trades: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Capture full error context for debugging"""

    context = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "stack_trace": traceback.format_exc(),
        "system_state": {
            "market_data_available": market_data is not None,
            "trades_count": len(trades) if trades else 0,
            "config_available": config is not None,
        }
    }

    # Add last 10 closes if market data available
    if market_data is not None and not market_data.empty:
        try:
            last_closes = market_data['close'].tail(10).tolist()
            context["market_data"] = {
                "last_10_closes": last_closes,
                "last_close": last_closes[-1] if last_closes else None,
                "bars_available": len(market_data)
            }
        except Exception:
            context["market_data"] = {"error": "Could not extract market data"}

    # Add recent trades
    if trades:
        try:
            context["recent_trades"] = trades[-10:]  # Last 10 trades
        except Exception:
            context["recent_trades"] = {"error": "Could not extract trades"}

    # Add configuration
    if config:
        context["configuration"] = config

    return context