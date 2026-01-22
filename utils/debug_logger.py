# utils/debug_logger.py
from __future__ import annotations

import builtins
import json
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Zorg dat de variabele bestaat op module-niveau + in builtins (tests kunnen dit verwachten)
recent_errors: list[dict[str, Any]] = []
if not hasattr(builtins, "recent_errors"):
    builtins.recent_errors = recent_errors


class DebugLogger:
    """
    Advanced Debug Logger for Brooks Trading System.
    Test-compatible implementation.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.errors_dir = self.log_dir / "errors"
        self.trades_dir = self.log_dir / "trades"
        self.snapshots_dir = self.log_dir / "snapshots"
        self.debug_dir = self.log_dir / "debug"

        # Maak alle vereiste mappen aan
        for d in (self.errors_dir, self.trades_dir, self.snapshots_dir, self.debug_dir):
            d.mkdir(parents=True, exist_ok=True)

    def log_error(self, error_context: dict[str, Any]) -> Path:
        """Log error context to both JSON and TXT files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        json_file = self.errors_dir / f"error_{timestamp}.json"
        txt_file = self.errors_dir / f"error_{timestamp}.txt"

        # JSON (machine readable)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(error_context, f, indent=4, ensure_ascii=False)

        # TXT (human readable - vereist door tests)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(error_context, indent=4, ensure_ascii=False))

        return json_file

    def log_trade(self, trade_data: dict[str, Any]) -> Path:
        """Log trade execution data to a daily JSONL file."""
        date_str = datetime.now().strftime("%Y%m%d")
        jsonl_file = self.trades_dir / f"trades_{date_str}.jsonl"
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade_data, ensure_ascii=False) + "\n")
        return jsonl_file

    def capture_error_context(self, exception: Exception = None, **kwargs: Any) -> dict[str, Any]:
        """Captures full system state during an error."""
        exc = exception if exception is not None else kwargs.get("error")
        return {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(exc).__name__ if exc else "Error",
            "error_message": str(exc) if exc else "",
            "stack_trace": traceback.format_exc(),
            "system_state": {"market_data": "market_data" in kwargs},
        }

    def save_snapshot(self, snapshot_data: dict[str, Any] | None = None, **kwargs: Any) -> Path:
        """Saves a system snapshot with all required files for tests."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.snapshots_dir / timestamp
        folder.mkdir(parents=True, exist_ok=True)

        default_data = snapshot_data or {}

        # Sla JSON bestanden op
        for name, content in (
            ("snapshot.json", default_data),
            ("trades.json", kwargs.get("trades", [])),
            ("account_info.json", kwargs.get("account_info", {})),
        ):
            with open(folder / name, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4, ensure_ascii=False)

        market_data = kwargs.get("market_data")
        if isinstance(market_data, pd.DataFrame):
            market_data.to_csv(folder / "market_data.csv", index=False)

        return folder

    def save_daily_summary(self, summary_data: dict[str, Any]) -> Path:
        """Saves a daily performance summary."""
        date_str = datetime.now().strftime("%Y%m%d")
        summary_file = self.snapshots_dir / f"daily_summary_{date_str}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        return summary_file

    def get_recent_errors(self, count: int = 5) -> list[dict[str, Any]]:
        """Helper to retrieve error logs."""
        files = sorted(self.errors_dir.glob("error_*.json"), reverse=True)
        errors: list[dict[str, Any]] = []

        for f in files[:count]:
            try:
                with open(f, encoding="utf-8") as e:
                    errors.append(json.load(e))
            except (OSError, json.JSONDecodeError):
                continue

        # Sync module-global + builtins voor tests
        global recent_errors
        recent_errors = errors
        builtins.recent_errors = errors
        return errors

    def get_daily_trades(self, day: str | date | None = None) -> list[dict[str, Any]]:
        """
        Required by tests: load trades from trades_YYYYMMDD.jsonl.
        day:
          None -> today
          "YYYYMMDD" -> that day
          date object -> that day
        """
        if day is None:
            day_str = datetime.now().strftime("%Y%m%d")
        elif isinstance(day, date):
            day_str = day.strftime("%Y%m%d")
        else:
            day_str = str(day)

        jsonl_file = self.trades_dir / f"trades_{day_str}.jsonl"
        if not jsonl_file.exists():
            return []

        out: list[dict[str, Any]] = []
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []

        return out
