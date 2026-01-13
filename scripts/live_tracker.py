# scripts/live_tracker.py
"""
Track live trading performance
"""
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


class LiveTracker:
    def __init__(self, log_file: str = "live_trades.json"):
        self.log_file = Path(log_file)
        self.trades = self.load_trades()

    def load_trades(self):
        """Load existing trades from JSON"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []

    def save_trades(self):
        """Save trades to JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)

    def log_trade(self, trade_data: dict):
        """Log a completed trade"""
        self.trades.append({
            **trade_data,
            "logged_at": datetime.now().isoformat()
        })
        self.save_trades()

    def get_stats(self):
        """Calculate live performance stats"""
        if not self.trades:
            return None

        df = pd.DataFrame(self.trades)
        results = df['result_r'].values

        equity = results.cumsum()
        running_max = pd.Series(equity).cummax()
        drawdown = equity - running_max

        return {
            "total_trades": len(results),
            "net_r": float(equity[-1]),
            "winrate": float((results > 0).sum() / len(results)),
            "expectancy": float(results.mean()),
            "max_dd": float(drawdown.min()),
            "current_dd": float(drawdown[-1]),
            "last_10_avg": float(results[-10:].mean()) if len(results) >= 10 else None,
        }

    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        if not stats:
            print("No trades yet")
            return

        print("\n" + "=" * 50)
        print("  LIVE PERFORMANCE")
        print("=" * 50)
        print(f"Trades      : {stats['total_trades']}")
        print(f"Net R       : {stats['net_r']:+.2f}R")
        print(f"Winrate     : {stats['winrate'] * 100:.1f}%")
        print(f"Expectancy  : {stats['expectancy']:+.4f}R")
        print(f"Max DD      : {stats['max_dd']:.2f}R")
        print(f"Current DD  : {stats['current_dd']:.2f}R")
        if stats['last_10_avg']:
            print(f"Last 10 avg : {stats['last_10_avg']:+.4f}R")
        print("=" * 50 + "\n")