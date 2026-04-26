from trader import (
    place_trade,
    print_account_info,
    print_position,
    check_stop_loss,
    check_trailing_stop,
    get_open_positions_count,
    get_total_market_value,
    already_holding,
    trading_client,
)

from strategy import scan_universe, wait_for_market_open
from universe import UNIVERSE

from config import (
    DOLLARS_PER_TRADE,
    MAX_POSITIONS,
    MAX_TOTAL_CAPITAL,
    STOP_LOSS_PERCENT,
    TRAILING_STOP_PERCENT,
    MA_SHORT,
    MA_LONG,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    SCAN_INTERVAL_SECONDS,
    MAX_CANDIDATES_PER_CYCLE
)

import time
import traceback


def run_bot():
    wait_for_market_open(trading_client)
    print("Starting trend scanner bot...")

    while True:
        print("\n==============================")
        print("NEW BOT CYCLE STARTING")
        print("==============================")

        # Manage exits first
        print("\n=== MANAGING OPEN POSITIONS ===")
        for symbol in UNIVERSE:
            try:
                stop_triggered = check_stop_loss(symbol, STOP_LOSS_PERCENT)
                trailing_triggered = check_trailing_stop(symbol, TRAILING_STOP_PERCENT)

                if already_holding(symbol):
                    print_position(symbol)

            except Exception as e:
                print(f"Error managing {symbol}: {e}")

        # Scan for new entries
        print("\n=== SCANNING FOR NEW ENTRIES ===")
        candidates = scan_universe(
            UNIVERSE,
            MA_SHORT,
            MA_LONG,
            MACD_FAST,
            MACD_SLOW,
            MACD_SIGNAL
        )

        print(f"Found {len(candidates)} candidates.")

        open_positions = get_open_positions_count()
        total_capital_used = get_total_market_value()

        for candidate in candidates[:MAX_CANDIDATES_PER_CYCLE]:
            symbol = candidate["symbol"]
            price = candidate["price"]

            if open_positions >= MAX_POSITIONS:
                print("Max positions reached.")
                break

            if total_capital_used + DOLLARS_PER_TRADE > MAX_TOTAL_CAPITAL:
                print("Max total capital reached.")
                break

            if already_holding(symbol):
                print(f"Already holding {symbol}. Skipping.")
                continue

            place_trade(symbol, "buy", notional=DOLLARS_PER_TRADE)
            time.sleep(3)

            open_positions = get_open_positions_count()
            total_capital_used = get_total_market_value()

        print_account_info()
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    while True:
        try:
            run_bot()
        except KeyboardInterrupt:
            print("\nBot stopped manually.")
            break
        except Exception as e:
            print("\nBOT CRASHED - restarting soon...")
            print(f"Crash reason: {e}")
            traceback.print_exc()
            time.sleep(30)
