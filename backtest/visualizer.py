# backtest/visualiser.py
"""
Enhanced visualization with parameter details and price chart
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from strategies.h2l2 import PlannedTrade, Side

logger = logging.getLogger(__name__)


def generate_performance_report(
        results_r: List[float],
        equity_curve: pd.Series,
        drawdown: pd.Series,
        symbol: str = "UNKNOWN",
        days: int = 0,
        m5_data: Optional[pd.DataFrame] = None,
        trades: Optional[List[PlannedTrade]] = None,
        config: Optional[Dict[str, Any]] = None,
):
    """
    Enhanced dashboard with:
    - Parameter details in title
    - Timestamp in filename
    - S&P 500 price chart with trade markers

    Args:
        results_r: List of trade results in R
        equity_curve: Cumulative equity curve
        drawdown: Drawdown series
        symbol: Symbol name
        days: Number of days backtested
        m5_data: M5 OHLC data (for price chart)
        trades: List of executed trades (for markers)
        config: Dict with backtest config (stop_buffer, cooldown, etc.)
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build config string for title
    config_str = ""
    if config:
        parts = []
        if config.get('regime_filter'):
            parts.append(f"Regime={config.get('chop_threshold', 2.5):.1f}")
        parts.append(f"StopBuf={config.get('stop_buffer', 2.0):.1f}")
        parts.append(f"Cool={config.get('cooldown_bars', 10)}")
        if config.get('costs_per_trade_r', 0) > 0:
            parts.append(f"Costs={config['costs_per_trade_r']:.2f}R")
        config_str = " | ".join(parts)

    # Determine number of subplots
    has_price_data = m5_data is not None and trades is not None
    n_plots = 5 if has_price_data else 4

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 18 if has_price_data else 14))

    if has_price_data:
        gs = fig.add_gridspec(5, 1, height_ratios=[2, 1.5, 1, 1, 1])
    else:
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # Equity
    ax_price_idx = 1 if has_price_data else None
    ax2 = fig.add_subplot(gs[ax_price_idx]) if has_price_data else None  # Price chart
    ax3 = fig.add_subplot(gs[2 if has_price_data else 1])  # Drawdown
    ax4 = fig.add_subplot(gs[3 if has_price_data else 2])  # Rolling Winrate
    ax5 = fig.add_subplot(gs[4 if has_price_data else 3])  # Trade Distribution

    # Title with config details
    title = f'Brooks MVP Dashboard: {symbol} ({days} Days)'
    if config_str:
        title += f'\n{config_str}'

    # 1. EQUITY CURVE
    ax1.plot(equity_curve.values, label='Equity Growth (R)', color='#2ca02c', linewidth=2.5)
    ax1.fill_between(range(len(equity_curve)), equity_curve, color='#2ca02c', alpha=0.1)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative R', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add final R value
    final_r = equity_curve.iloc[-1]
    ax1.text(0.98, 0.95, f'Final: {final_r:+.1f}R',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='green'))

    # 2. PRICE CHART (if data provided)
    if has_price_data and ax2 is not None:
        # Resample to daily for cleaner visualization
        daily_data = m5_data.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        dates = daily_data.index
        closes = daily_data['close']

        # Plot price
        ax2.plot(dates, closes, color='#1f77b4', linewidth=1.5, label=f'{symbol} Close', alpha=0.8)
        ax2.fill_between(dates, daily_data['low'], daily_data['high'],
                         color='#1f77b4', alpha=0.1, label='Daily Range')

        # Mark trades on price chart
        for i, trade in enumerate(trades):
            if i >= len(results_r):
                break

            result = results_r[i]
            color = '#2ca02c' if result > 0 else '#d62728'
            marker = '^' if trade.side == Side.LONG else 'v'
            size = 100 if abs(result) > 1.5 else 50

            # Find trade date
            trade_date = trade.execute_ts
            if trade_date in daily_data.index:
                price = daily_data.loc[trade_date, 'close']
                ax2.scatter(trade_date, price,
                            color=color, marker=marker, s=size, alpha=0.6,
                            edgecolors='black', linewidths=0.8, zorder=5)

        ax2.set_ylabel(f'{symbol} Price', fontsize=11)
        ax2.set_xlabel('')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

        # Add period info
        start_date = dates[0].strftime('%Y-%m-%d')
        end_date = dates[-1].strftime('%Y-%m-%d')
        ax2.text(0.02, 0.95, f'Period: {start_date} to {end_date}',
                 transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round',
                                                    facecolor='white', alpha=0.7))

    # 3. DRAWDOWN CHART
    ax3.fill_between(range(len(drawdown)), drawdown, 0, color='#d62728', alpha=0.3)
    ax3.plot(drawdown.values, color='#d62728', linewidth=1.5, label='Drawdown (R)')
    ax3.set_ylabel('Drawdown (R)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left', fontsize=10)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Max DD marker
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    ax3.scatter(max_dd_idx, max_dd, color='red', s=100, zorder=5,
                edgecolors='black', linewidths=1.5)
    ax3.text(max_dd_idx, max_dd, f' Max: {max_dd:.1f}R',
             fontsize=9, verticalalignment='top')

    # 4. ROLLING WINRATE
    res_series = pd.Series(results_r)
    rolling_winrate = (res_series > 0).rolling(window=30, min_periods=10).mean() * 100

    ax4.plot(rolling_winrate.values, color='#1f77b4', linewidth=2,
             label='Winrate % (Rolling 30)')
    ax4.axhline(33.3, color='orange', linestyle='--', linewidth=1.5,
                label='Breakeven (1:2 RR)', alpha=0.8)
    ax4.set_ylabel('Winrate %', fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=10)

    # Overall winrate
    overall_wr = (sum(1 for r in results_r if r > 0) / len(results_r) * 100) if results_r else 0
    ax4.text(0.98, 0.95, f'Overall: {overall_wr:.1f}%',
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 5. INDIVIDUAL TRADE RETURNS
    colors = ['#2ca02c' if r > 0 else '#d62728' for r in res_series]
    ax5.bar(range(len(res_series)), res_series, color=colors, alpha=0.6, width=1.0)
    ax5.set_ylabel('Trade Result (R)', fontsize=11)
    ax5.set_xlabel('Trade Number', fontsize=11)
    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.grid(True, alpha=0.3, axis='y')

    # Stats
    winners = sum(1 for r in res_series if r > 0)
    losers = sum(1 for r in res_series if r < 0)
    ax5.text(0.02, 0.95, f'W: {winners} | L: {losers}',
             transform=ax5.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save with timestamp
    filename = f"backtest_report_{symbol}_{days}d_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Dashboard '{filename}' saved with config: {config_str}")
    print(f"\n📊 Dashboard saved: {filename}")
    return filename