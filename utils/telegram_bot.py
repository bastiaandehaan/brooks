# utils/telegram_bot.py
"""
Telegram notifications for Brooks trading system
Now with .env support for security
"""
import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try to load .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")


@dataclass
class TradingSignal:
    """Structured trading signal data for Telegram notifications"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry: float
    stop: float
    target: float
    lots: float
    risk_usd: float
    risk_pct: float
    reason: str
    regime: Optional[str] = None


class TelegramBot:
    """
    Send trading notifications to Telegram

    Setup:
    1. Message @BotFather on Telegram
    2. Create new bot: /newbot
    3. Get your bot_token
    4. Message @userinfobot to get your chat_id
    5. Create .env file with credentials
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Args:
            bot_token: Bot token from @BotFather (or from .env)
            chat_id: Your chat ID from @userinfobot (or from .env)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token:
            raise ValueError(
                "❌ TELEGRAM_BOT_TOKEN not found!\n"
                "   Set it in .env file or pass to __init__"
            )

        if not self.chat_id:
            raise ValueError(
                "❌ TELEGRAM_CHAT_ID not found!\n"
                "   Set it in .env file or pass to __init__"
            )

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

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

Your Telegram bot is configured correctly!

✅ Daily Sharpe: 1.817 (Institutional Grade!)
✅ Annual Return: 41.41%
✅ Ready for live monitoring

<i>Test sent at {}</i>
        """.strip().format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        result = self.send_message(msg)

        if result:
            print("✅ Telegram bot test successful!")
            print("   Check your Telegram app for the test message.")
            return True
        else:
            print("❌ Telegram bot test failed!")
            print("   Check your bot_token and chat_id in .env")
            return False


# Test script
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 TESTING TELEGRAM BOT CONNECTION")
    print("=" * 60)

    try:
        # Create bot (reads from .env)
        bot = TelegramBot()

        print("\n✅ Bot initialized successfully")
        print(f"   Bot Token: {bot.bot_token[:20]}...")
        print(f"   Chat ID: {bot.chat_id}")

        print("\n📤 Sending test message...")
        success = bot.test_connection()

        if success:
            print("\n" + "=" * 60)
            print("✅ SUCCESS! Check your Telegram for the test message")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Verify you received the message in Telegram")
            print("2. Test live_monitor.py during NY session (14:30-21:00 CET)")
            print("3. Start paper trading!")
        else:
            print("\n" + "=" * 60)
            print("❌ TEST FAILED")
            print("=" * 60)
            print("\nTroubleshooting:")
            print("1. Check .env file exists in project root")
            print("2. Verify TELEGRAM_BOT_TOKEN is correct")
            print("3. Verify TELEGRAM_CHAT_ID is correct")
            print("4. Start a chat with your bot (search bot username in Telegram)")

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nSetup Instructions:")
        print("1. Create .env file in project root")
        print("2. Add these lines:")
        print("   TELEGRAM_BOT_TOKEN=8597453018:AAHs30mJkqs64BbTgIg6L7npW1Q3f5HbVPw")
        print("   TELEGRAM_CHAT_ID=6156828622")
        print("3. Save the file")
        print("4. Run this script again")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")