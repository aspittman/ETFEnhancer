from trader import place_trade, print_account_info, print_position, check_stop_loss
import time
from config import SYMBOLS, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT, TRADE_SIZE, STOP_LOSS_PERCENT
from strategy import check_signal
from trader import place_trade
from strategy import wait_for_market_open
from trader import trading_client
from trader import place_trade, print_account_info, print_position
from trader import check_trailing_stop
from trader import get_open_positions_count
from config import TRAILING_STOP_PERCENT, MAX_POSITIONS

wait_for_market_open(trading_client)

print("Starting Alpaca Paper Trading Bot...")

last_action = None

while True:
    for symbol in SYMBOLS:
        print(f"\n=== Checking {symbol} ===")

        signal = check_signal(
            symbol,
            RSI_PERIOD,
            RSI_OVERSOLD,
            RSI_OVERBOUGHT
        )

        print(f"{symbol} Signal: {signal}")

        stop_triggered = check_stop_loss(symbol, STOP_LOSS_PERCENT)
        trailing_triggered = check_trailing_stop(symbol, TRAILING_STOP_PERCENT)

        if not stop_triggered and not trailing_triggered:
            if signal in ["buy", "sell"]:
                if get_open_positions_count() < MAX_POSITIONS:
                    place_trade(symbol, signal, TRADE_SIZE)
                    time.sleep(3)
                else:
                    print("Max positions reached. Skipping trade.")

        print_position(symbol)

    print_account_info()

    time.sleep(300)
