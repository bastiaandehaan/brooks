# utils/telegram_bot.py
"""
Telegram notifications for Brooks trading system
"""
import requests
from datetime import datetime
from typing import Optional, Dict, Any


class TelegramBot:
    """
    Send trading notifications to Telegram

    Setup:
    1. Message @BotFather on Telegram
    2. Create new bot: /newbot
    3. Get your bot_token
    4. Message @userinfobot to get your chat_id
    5. Start conversation with your bot
    """

    def __init__(self, bot_token: str, chat_id: str):
        """
        Args:
            bot_token: Bot token from @BotFather
            chat_id: Your chat ID from @userinfobot
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, text: str, parse_mode: str = "HTML") -> Optional[Dict]:
        """
        Send text message

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"

        Returns:
            Response dict or None if failed
        """
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            return None

    def send_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Send trade signal notification

        Args:
            signal: Dict with keys: trade, lots, risk_usd, balance

        Returns:
            True if sent successfully
        """
        trade = signal['trade']
        lots = signal['lots']
        risk_usd = signal['risk_usd']

        # Direction emoji
        emoji = "🟢" if trade.side.value == "LONG" else "🔴"

        msg = f"""
{emoji} <b>BROOKS SIGNAL DETECTED</b>

<b>Direction:</b> {trade.side.value}
<b>Entry:</b> {trade.entry:.2f}
<b>Stop Loss:</b> {trade.stop:.2f}
<b>Take Profit:</b> {trade.tp:.2f}

<b>Position Size:</b> {lots:.2f} lots
<b>Risk:</b> ${risk_usd:.2f} ({(risk_usd / signal['balance'] * 100):.2f}%)

<b>Setup:</b> {trade.reason}
<b>Time:</b> {trade.signal_ts.strftime('%H:%M:%S')} UTC

⚠️ <i>Manually enter this trade in your platform</i>
        """.strip()

        result = self.send_message(msg)
        return result is not None

    def send_fill(self, side: str, entry: float, lots: float, sl: float, tp: float) -> bool:
        """
        Confirm trade was executed

        Args:
            side: "LONG" or "SHORT"
            entry: Entry price
            lots: Position size
            sl: Stop loss
            tp: Take profit

        Returns:
            True if sent successfully
        """
        emoji = "✅"

        msg = f"""
{emoji} <b>TRADE ENTERED</b>

<b>{side}</b> {lots:.2f} lots @ {entry:.2f}
SL: {sl:.2f} | TP: {tp:.2f}

<i>Position is now active</i>
        """.strip()

        result = self.send_message(msg)
        return result is not None

    def send_exit(
            self,
            side: str,
            entry: float,
            exit_price: float,
            result_r: float,
            pnl_usd: float
    ) -> bool:
        """
        Report trade exit

        Args:
            side: "LONG" or "SHORT"
            entry: Entry price
            exit_price: Exit price
            result_r: Result in R (e.g., +2.0 or -1.0)
            pnl_usd: Profit/loss in USD

        Returns:
            True if sent successfully
        """
        emoji = "🟢" if result_r > 0 else "🔴"
        outcome = "WIN" if result_r > 0 else "LOSS"

        msg = f"""
{emoji} <b>TRADE CLOSED - {outcome}</b>

<b>{side}</b>
Entry: {entry:.2f} → Exit: {exit_price:.2f}

<b>Result:</b> {result_r:+.2f}R (${pnl_usd:+.2f})

{self._get_motivational_message(result_r)}
        """.strip()

        result = self.send_message(msg)
        return result is not None

    def send_daily_summary(self, stats: Dict[str, Any]) -> bool:
        """
        Send end-of-day summary

        Args:
            stats: Dict with keys: trades_today, net_r_today, total_r, total_trades

        Returns:
            True if sent successfully
        """
        msg = f"""
📊 <b>DAILY SUMMARY</b>

<b>Today:</b>
  • Trades: {stats.get('trades_today', 0)}
  • Net R: {stats.get('net_r_today', 0):+.2f}R

<b>Total (All Time):</b>
  • Trades: {stats.get('total_trades', 0)}
  • Net R: {stats.get('total_r', 0):+.2f}R
  • Winrate: {stats.get('winrate', 0) * 100:.1f}%

<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>
        """.strip()

        result = self.send_message(msg)
        return result is not None

    def send_error(self, error_msg: str) -> bool:
        """
        Send error notification

        Args:
            error_msg: Error description

        Returns:
            True if sent successfully
        """
        msg = f"""
⚠️ <b>SYSTEM ERROR</b>

{error_msg}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """.strip()

        result = self.send_message(msg)
        return result is not None

    def _get_motivational_message(self, result_r: float) -> str:
        """Get a motivational message based on result"""
        if result_r >= 2.0:
            return "🎯 Perfect execution! Target hit!"
        elif result_r > 0:
            return "✨ Profit is profit! Well done."
        elif result_r >= -0.5:
            return "💪 Small loss, keep going!"
        else:
            return "📚 Every loss is a lesson. Stay disciplined!"

    def test_connection(self) -> bool:
        """
        Test if bot is working

        Returns:
            True if test message sent successfully
        """
        msg = """
🤖 <b>BROOKS BOT TEST</b>

If you see this message, your Telegram bot is configured correctly!

<i>Test sent at {}</i>
        """.strip().format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        result = self.send_message(msg)

        if result:
            print("✅ Telegram bot test successful!")
            return True
        else:
            print("❌ Telegram bot test failed!")
            return False


# Example usage
if __name__ == "__main__":
    # CONFIGURATION
    # Replace these with your actual values:
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # From @BotFather
    CHAT_ID = "YOUR_CHAT_ID_HERE"  # From @userinfobot

    # Create bot
    bot = TelegramBot(bot_token=BOT_TOKEN, chat_id=CHAT_ID)

    # Test connection
    print("Testing Telegram bot...")
    bot.test_connection()

    # Example signal
    # from strategies.h2l2 import PlannedTrade, Side
    # import pandas as pd
    #
    # signal = {
    #     'trade': PlannedTrade(
    #         signal_ts=pd.Timestamp.now(tz='UTC'),
    #         execute_ts=pd.Timestamp.now(tz='UTC'),
    #         side=Side.LONG,
    #         entry=6010.0,
    #         stop=6000.0,
    #         tp=6030.0,
    #         reason="H2 LONG: rejection after 3bar swing"
    #     ),
    #     'lots': 0.1,
    #     'risk_usd': 50.0,
    #     'balance': 10000.0,
    # }
    #
    # bot.send_signal(signal)