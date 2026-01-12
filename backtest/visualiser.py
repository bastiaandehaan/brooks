# backtest/visualiser.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_performance_report(results_r, equity_curve, drawdown, symbol=None, days=None):
    """
    Uitgebreid dashboard met Equity, Drawdown en Rolling Winrate.
    Backward compatible: symbol/days zijn optioneel.
    """
    symbol = symbol or "UNKNOWN"
    days = days or "NA"

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # Equity
    ax2 = fig.add_subplot(gs[1])  # Drawdown
    ax3 = fig.add_subplot(gs[2])  # Rolling Winrate
    ax4 = fig.add_subplot(gs[3])  # Distribution

    # 1. Equity Curve
    ax1.plot(equity_curve.values, label='Equity Growth (R)', color='#2ca02c', linewidth=2.5)
    ax1.fill_between(range(len(equity_curve)), equity_curve, color='#2ca02c', alpha=0.1)
    ax1.set_title(f'Brooks MVP Dashboard: {symbol} ({days} Days)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative R')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 2. Drawdown Chart
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='#d62728', alpha=0.3)
    ax2.plot(drawdown.values, color='#d62728', linewidth=1, label='Drawdown (R)')
    ax2.set_ylabel('Drawdown (R)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')

    # 3. Rolling Winrate (per 30 trades)
    res_series = pd.Series(results_r)
    rolling_winrate = (res_series > 0).rolling(window=30).mean() * 100
    ax3.plot(rolling_winrate.values, color='#1f77b4', linewidth=2, label='Winrate % (Rolling 30)')
    ax3.axhline(33.3, color='orange', linestyle='--', label='Breakeven (1:2 RR)')
    ax3.set_ylabel('Winrate %')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')

    # 4. Individual Trade Returns
    colors = ['#2ca02c' if r > 0 else '#d62728' for r in res_series]
    ax4.bar(range(len(res_series)), res_series, color=colors, alpha=0.6)
    ax4.set_ylabel('Trade Result (R)')
    ax4.set_xlabel('Trade Count')
    ax4.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()

    filename = f"backtest_report_{symbol}_{days}d.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Dashboard '{filename}' opgeslagen.")
