from trader import (
    place_trade,
    print_account_info,
    print_position,
    check_stop_loss,
    check_atr_trailing_stop,
    check_dynamic_midpoint_stop,
    get_open_positions_count,
    get_total_market_value,
    already_holding,
    trading_client,
    validate_alpaca_credentials,
    AlpacaAuthError,
)

from strategy import StrategySwitches, get_market_regime, scan_universe, wait_for_market_open
from universe import UNIVERSE
from trader import is_in_cooldown

from config import (
    DOLLARS_PER_TRADE,
    MAX_POSITIONS,
    MAX_TOTAL_CAPITAL,
    STOP_LOSS_PERCENT,
    ATR_TRAILING_MULTIPLIER,
    ATR_WINDOW,
    MA_SHORT,
    MA_LONG,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    SCAN_INTERVAL_SECONDS,
    MAX_CANDIDATES_PER_CYCLE,
    MARKET_REGIME_SYMBOL,
    MARKET_REGIME_MA_SHORT,
    MARKET_REGIME_MA_LONG,
    MIN_CANDIDATE_SCORE,
    MAX_NEW_BUYS_PER_CYCLE,
    BLOCKED_SYMBOLS,
    ENABLE_ATR_TRAILING_STOP,
    ENABLE_DYNAMIC_MIDPOINT_STOP,
    ENABLE_ATR_TREND_FILTER,
    ENABLE_FIXED_STOP_LOSS,
    ENABLE_MACD_FILTER,
    ENABLE_MARKET_REGIME_FILTER,
    ENABLE_MA_ALIGNMENT_FILTER,
    ENABLE_MOMENTUM_SCORE,
    ENABLE_MIN_SCORE_FILTER,
    ENABLE_RELATIVE_STRENGTH_FILTER,
    ENABLE_TOP_CANDIDATE_SELECTION,
    ENABLE_VOLUME_FILTER,
    ENABLE_BULLISH_CANDLE_CONFIRMATION,
)

import time
import traceback


def run_bot():
    validate_alpaca_credentials()
    wait_for_market_open(trading_client)
    print("Starting trend scanner bot...")
    switches = StrategySwitches(
        market_regime=ENABLE_MARKET_REGIME_FILTER,
        ma_alignment=ENABLE_MA_ALIGNMENT_FILTER,
        macd=ENABLE_MACD_FILTER,
        relative_strength=ENABLE_RELATIVE_STRENGTH_FILTER,
        price_momentum=ENABLE_MOMENTUM_SCORE,
        volume=ENABLE_VOLUME_FILTER,
        atr_trend_quality=ENABLE_ATR_TREND_FILTER,
        min_score=ENABLE_MIN_SCORE_FILTER and MIN_CANDIDATE_SCORE is not None,
        top_candidates=ENABLE_TOP_CANDIDATE_SELECTION,
        bullish_candle=ENABLE_BULLISH_CANDLE_CONFIRMATION,
    )

    while True:
        print("\n==============================")
        print("NEW BOT CYCLE STARTING")
        print("==============================")

        # Manage exits first
        print("\n=== MANAGING OPEN POSITIONS ===")
        for symbol in UNIVERSE:
            try:
                if ENABLE_FIXED_STOP_LOSS:
                    check_stop_loss(symbol, STOP_LOSS_PERCENT)
                if ENABLE_ATR_TRAILING_STOP:
                    check_atr_trailing_stop(symbol, ATR_TRAILING_MULTIPLIER)
                if ENABLE_DYNAMIC_MIDPOINT_STOP:
                    check_dynamic_midpoint_stop(symbol)

                if already_holding(symbol):
                    print_position(symbol)

            except Exception as e:
                if isinstance(e, AlpacaAuthError):
                    raise
                print(f"Error managing {symbol}: {e}")

        # Scan for new entries
        print("\n=== SCANNING FOR NEW ENTRIES ===")
        if switches.market_regime:
            regime = get_market_regime(
                MARKET_REGIME_SYMBOL,
                MARKET_REGIME_MA_SHORT,
                MARKET_REGIME_MA_LONG,
            )
            print(f"Market regime: {regime}")
            if not regime["is_healthy"]:
                print("Market regime is weak. Skipping new buys this cycle.")
                print_account_info()
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

        candidates = scan_universe(
            UNIVERSE,
            MA_SHORT,
            MA_LONG,
            MACD_FAST,
            MACD_SLOW,
            MACD_SIGNAL,
            ATR_WINDOW,
            MIN_CANDIDATE_SCORE,
            MAX_CANDIDATES_PER_CYCLE,
            BLOCKED_SYMBOLS,
            MARKET_REGIME_SYMBOL,
            switches,
        )

        print(f"Found {len(candidates)} candidates.")
        for rank, candidate in enumerate(candidates[:MAX_CANDIDATES_PER_CYCLE], start=1):
            print(
                f"#{rank} {candidate['symbol']} "
                f"score={candidate['score']:.2f} "
                f"price={candidate['price']:.2f} "
                f"atr={candidate['atr']:.2f}"
            )

        open_positions = get_open_positions_count()
        total_capital_used = get_total_market_value()
        buys_this_cycle = 0

        for candidate in candidates[:MAX_CANDIDATES_PER_CYCLE]:
            symbol = candidate["symbol"]
            price = candidate["price"]

            if buys_this_cycle >= MAX_NEW_BUYS_PER_CYCLE:
                print("Max buys for this cycle reached.")
                break

            if open_positions >= MAX_POSITIONS:
                print("Max positions reached.")
                break

            if total_capital_used + DOLLARS_PER_TRADE > MAX_TOTAL_CAPITAL:
                print("Max total capital reached.")
                break

            if already_holding(symbol):
                print(f"Already holding {symbol}. Skipping.")
                continue

            if is_in_cooldown(symbol):
                continue
            
            place_trade(symbol, "buy", notional=DOLLARS_PER_TRADE)
            buys_this_cycle += 1
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
        except AlpacaAuthError as e:
            print("\nBOT STOPPED - Alpaca authentication failed.")
            print(e)
            break
        except Exception as e:
            print("\nBOT CRASHED - restarting soon...")
            print(f"Crash reason: {e}")
            traceback.print_exc()
            time.sleep(30)
