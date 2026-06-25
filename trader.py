from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from strategy import build_strategy_frame, fetch_price_history, wait_for_market_open
from config import (
    API_KEY,
    SECRET_KEY,
    ALPACA_PAPER,
    ATR_WINDOW,
    MA_SHORT,
    MA_LONG,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ATR_TRAILING_MULTIPLIER,
)
import csv
import os
import time
from datetime import datetime

highest_price = {}
recently_sold = {}

COOLDOWN_SECONDS = 3600  # 1 hour
LOG_FILE = "logs/trades.csv"

def mark_recently_sold(symbol):
    recently_sold[symbol] = time.time()


def is_in_cooldown(symbol):
    if symbol not in recently_sold:
        return False

    elapsed = time.time() - recently_sold[symbol]

    if elapsed < COOLDOWN_SECONDS:
        remaining = int((COOLDOWN_SECONDS - elapsed) / 60)
        print(f"{symbol} is cooling down for {remaining} more minutes.")
        return True

    recently_sold.pop(symbol, None)
    return False

def log_trade(symbol, side, qty, price, reason, entry_price=None, exit_price=None, pnl=None):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "symbol",
                "side",
                "qty",
                "price",
                "reason",
                "entry_price",
                "exit_price",
                "pnl"
            ])

        writer.writerow([
            datetime.now(),
            symbol,
            side,
            qty,
            price,
            reason,
            entry_price,
            exit_price,
            pnl
        ])
        
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=ALPACA_PAPER)

def get_total_market_value():
    try:
        positions = trading_client.get_all_positions()
        total = sum(float(p.market_value) for p in positions)
        return total
    except Exception as e:
        print(f"Error getting total market value: {e}")
        return 0.0

from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

def has_open_order(symbol):
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = trading_client.get_orders(filter=request)

        for order in orders:
            if order.symbol == symbol:
                return True

    except Exception as e:
        print(f"Error checking open orders: {e}")

    return False

def already_holding(symbol):
    return get_position(symbol) > 0

def get_open_positions_count():
    positions = trading_client.get_all_positions()
    return len(positions)

def get_latest_atr(symbol):
    data = fetch_price_history(symbol, period="1y", interval="1h")
    if data.empty:
        return None

    frame = build_strategy_frame(
        data,
        MA_SHORT,
        MA_LONG,
        MACD_FAST,
        MACD_SLOW,
        MACD_SIGNAL,
        ATR_WINDOW,
    )
    if frame.empty or "atr" not in frame.columns:
        return None

    latest_atr = frame["atr"].dropna()
    if latest_atr.empty:
        return None

    return float(latest_atr.iloc[-1])


def check_atr_trailing_stop(symbol, atr_multiplier):
    if has_open_order(symbol):
        print(f"Sell order already pending for {symbol}")
        return True
    
    try:
        position = trading_client.get_open_position(symbol)

        entry_price = float(position.avg_entry_price)
        current_price = float(position.current_price)

        # Initialize highest price
        if symbol not in highest_price:
            highest_price[symbol] = entry_price

        # Update highest price
        if current_price > highest_price[symbol]:
            highest_price[symbol] = current_price

        latest_atr = get_latest_atr(symbol)
        if latest_atr is None or latest_atr <= 0:
            print(f"ATR unavailable for {symbol}. Skipping ATR trailing stop.")
            return False

        trail_stop_price = highest_price[symbol] - (latest_atr * atr_multiplier)

        print(f"Highest Price: {highest_price[symbol]}")
        print(f"ATR: {latest_atr}")
        print(f"ATR Trailing Stop Price: {trail_stop_price}")

        if current_price <= trail_stop_price:
            print("ATR TRAILING STOP TRIGGERED!")

            qty = float(position.qty)
            pnl = (current_price - entry_price) * qty

            log_trade(
                symbol=symbol,
                side="sell",
                qty=qty,
                price=current_price,
                reason="atr_trailing_stop",
                entry_price=entry_price,
                exit_price=current_price,
                pnl=pnl
            )

            place_trade(symbol, "sell", qty=qty, reason="atr_trailing_stop")
            mark_recently_sold(symbol)
            highest_price.pop(symbol, None)

            return True

    except:
        return False

    return False


def check_trailing_stop(symbol, trailing_percent):
    print("Fixed percent trailing stops are deprecated; using ATR trailing stop.")
    return check_atr_trailing_stop(symbol, ATR_TRAILING_MULTIPLIER)

def get_position(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except APIError:
        return 0

def check_stop_loss(symbol, stop_loss_percent):
    if has_open_order(symbol):
        print(f"Sell order already pending for {symbol}")
        return True
    
    try:
        position = trading_client.get_open_position(symbol)

        entry_price = float(position.avg_entry_price)
        current_price = float(position.current_price)

        loss_threshold = entry_price * (1 - stop_loss_percent)

        print(f"Entry Price: {entry_price}")
        print(f"Current Price: {current_price}")
        print(f"Stop Loss Price: {loss_threshold}")

        if current_price <= loss_threshold:
            print("STOP LOSS TRIGGERED!")

            qty = float(position.qty)
            pnl = (current_price - entry_price) * qty

            log_trade(
                symbol=symbol,
                side="sell",
                qty=qty,
                price=current_price,
                reason="stop_loss",
                entry_price=entry_price,
                exit_price=current_price,
                pnl=pnl
            )

            place_trade(symbol, "sell", qty=qty, reason="stop_loss")
            mark_recently_sold(symbol)
            return True

    except:
        return False

    return False

def place_trade(symbol, side, qty=None, notional=15, reason="signal"):
    current_position = get_position(symbol)

    if has_open_order(symbol):
        print(f"Open order exists for {symbol}. Skipping...")
        return

    if side == "buy" and current_position > 0:
        print("Already holding position. Skipping buy.")
        return

    if side == "sell" and current_position == 0:
        print("No shares to sell. Skipping sell.")
        return

    if side == "buy":
        order = MarketOrderRequest(
            symbol=symbol,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

    else:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

    try:
        trading_client.submit_order(order)

        if side == "buy":
            print(f"Placed BUY order for ${notional} of {symbol}")

            try:
                position = trading_client.get_open_position(symbol)
                entry_price = float(position.avg_entry_price)
                qty = float(position.qty)
            except:
                entry_price = None
                qty = None

            log_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                price=entry_price,
                reason=reason,
                entry_price=entry_price,
                exit_price=None,
                pnl=None
            )

        else:
            print(f"Placed SELL order for {qty} shares of {symbol}")

    except Exception as e:
        print(f"Order failed: {e}")

def print_account_info():
    account = trading_client.get_account()

    print("\n===== ACCOUNT INFO =====")
    print(f"Equity: ${account.equity}")
    print(f"Buying Power: ${account.buying_power}")
    print("========================\n")
    
def print_position(symbol):
    try:
        position = trading_client.get_open_position(symbol)

        print("----- POSITION -----")
        print(f"Symbol: {symbol}")
        print(f"Qty: {position.qty}")
        print(f"Avg Entry: ${position.avg_entry_price}")
        print(f"Current Price: ${position.current_price}")
        print(f"Unrealized P/L: ${position.unrealized_pl}")
        print("--------------------\n")

    except:
        print("No open position.\n")
