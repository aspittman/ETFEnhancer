from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from strategy import wait_for_market_open
from config import API_KEY, SECRET_KEY

highest_price = {}

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def calculate_qty(symbol, dollars_per_trade):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from config import API_KEY, SECRET_KEY

        data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(request)

        ask_price = float(quote[symbol].ask_price)

        if ask_price <= 0:
            print(f"Invalid ask price for {symbol}")
            return 0

        qty = int(dollars_per_trade // ask_price)
        return max(qty, 0)

    except Exception as e:
        print(f"Error calculating qty for {symbol}: {e}")
        return 0
    
def get_total_market_value():
    try:
        positions = trading_client.get_all_positions()
        total = sum(float(p.market_value) for p in positions)
        return total
    except Exception as e:
        print(f"Error getting total market value: {e}")
        return 0.0


def already_holding(symbol):
    return get_position(symbol) > 0

def get_open_positions_count():
    positions = trading_client.get_all_positions()
    return len(positions)

def check_trailing_stop(symbol, trailing_percent):
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

        trail_stop_price = highest_price[symbol] * (1 - trailing_percent)

        print(f"Highest Price: {highest_price[symbol]}")
        print(f"Trailing Stop Price: {trail_stop_price}")

        if current_price <= trail_stop_price:
            print("TRAILING STOP TRIGGERED!")
            place_trade(symbol, "sell", int(position.qty))

            # Reset after selling
            highest_price.pop(symbol, None)

            return True

    except:
        return False

    return False

def get_position(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return int(position.qty)
    except APIError:
        return 0

def check_stop_loss(symbol, stop_loss_percent):
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
            place_trade(symbol, "sell", int(position.qty))
            return True

    except:
        return False

    return False

def place_trade(symbol, side, qty):
    current_position = get_position(symbol)

    if side == "buy" and current_position > 0:
        print("Already holding position. Skipping buy.")
        return

    if side == "sell" and current_position == 0:
        print("No shares to sell. Skipping sell.")
        return

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    trading_client.submit_order(order)
    print(f"Placed {side.upper()} order for {qty} shares of {symbol}")

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