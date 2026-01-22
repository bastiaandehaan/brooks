# execution/order_manager.py
"""
Order execution and management
"""

from dataclasses import dataclass


@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    price: float
    message: str


class OrderManager:
    def __init__(self, mt5_module):
        self.mt5 = mt5_module

    def place_market_order(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        lots: float,
        sl: float,
        tp: float,
        comment: str = "Brooks",
    ) -> OrderResult:
        """Place market order with SL/TP"""

        # Get current price
        tick = self.mt5.symbol_info_tick(symbol)
        if not tick:
            return OrderResult(False, None, 0.0, "Failed to get tick")

        # Determine order type and price
        if side == "LONG":
            order_type = self.mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Build request
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,  # max slippage in points
            "magic": 777,  # Your unique ID
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = self.mt5.order_send(request)

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                ticket=None,
                price=0.0,
                message=f"Order failed: {result.retcode} - {result.comment}",
            )

        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            message="Order filled successfully",
        )

    def get_open_positions(self, symbol: str = None):
        """Get all open positions"""
        positions = self.mt5.positions_get(symbol=symbol)
        return list(positions) if positions else []

    def close_position(self, ticket: int):
        """Close position by ticket"""
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        position = positions[0]

        # Opposite order type
        if position.type == self.mt5.POSITION_TYPE_BUY:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(position.symbol).ask

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 777,
            "comment": "Close by script",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)
        return result.retcode == self.mt5.TRADE_RETCODE_DONE
