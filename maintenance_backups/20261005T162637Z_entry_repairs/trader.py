from bot_ownership import TradingClient
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
    PIVOT_LOOKBACK_WEEKS,
    PIVOT_REVERSAL_PERCENT,
    MIN_WEEKS_BETWEEN_PIVOTS,
    REQUIRE_CLOSE_BELOW_STRUCTURAL_STOP,
    STRUCTURAL_STOP_EXIT_TIMEFRAME,
    USE_TENTATIVE_HIGH_FOR_STOP,
    PIVOT_TIMEFRAME,
    PIVOT_PRICE_SOURCE,
    PERFORMANCE_START_DATE,
)
from pivots import (
    build_pivot_history,
    completed_weekly_closes,
    new_pivot_state,
    update_pivot_state,
    update_structural_stop,
)
from universe import LIVE_TRADING_UNIVERSE, is_live_trading_symbol
import csv
import json
import os
import time
import uuid
from datetime import date, datetime
import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout

highest_price = {}
recently_sold = {}
position_state = {}
pivot_state = {}

COOLDOWN_SECONDS = 3600  # 1 hour
TRADING_ENVIRONMENT = "paper" if ALPACA_PAPER else "live"
LOG_FILE = f"logs/trades_{TRADING_ENVIRONMENT}.csv"
POSITION_STATE_FILE = f"logs/position_state_{TRADING_ENVIRONMENT}.json"
PIVOT_STATE_FILE = f"logs/pivot_state_{TRADING_ENVIRONMENT}.json"
ALPACA_READ_RETRIES = 3
ALPACA_RETRY_DELAY_SECONDS = 2
ORDER_FILL_ATTEMPTS = 30
ORDER_FILL_DELAY_SECONDS = 1
ORDER_CLIENT_ID_PREFIX = f"etfenhancer-{TRADING_ENVIRONMENT}"


class AlpacaAuthError(RuntimeError):
    pass


def load_position_state():
    try:
        with open(POSITION_STATE_FILE) as file:
            loaded_state = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        loaded_state = {}
    # Keep the object identity stable so callers and tests holding a reference
    # always see the freshly loaded state.
    position_state.clear()
    position_state.update(loaded_state)
    return position_state


def load_pivot_state():
    global pivot_state
    try:
        with open(PIVOT_STATE_FILE) as file:
            pivot_state = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        pivot_state = {}
    return pivot_state


def save_position_state():
    os.makedirs(os.path.dirname(POSITION_STATE_FILE), exist_ok=True)
    temporary = POSITION_STATE_FILE + ".tmp"
    with open(temporary, "w") as file:
        json.dump(position_state, file, indent=2, sort_keys=True)
    os.replace(temporary, POSITION_STATE_FILE)


def save_pivot_state():
    os.makedirs(os.path.dirname(PIVOT_STATE_FILE), exist_ok=True)
    temporary = PIVOT_STATE_FILE + ".tmp"
    with open(temporary, "w") as file:
        json.dump(pivot_state, file, indent=2, sort_keys=True)
    os.replace(temporary, PIVOT_STATE_FILE)


load_position_state()
load_pivot_state()


def update_midpoint_state(state, entry_price, observed_price):
    old_high = float(state.get("highest_price_since_entry", entry_price))
    old_stop = float(state.get("current_midpoint_stop", entry_price))
    state["entry_price"] = entry_price
    state.setdefault("previous_high", old_high)
    if observed_price > old_high:
        state["previous_high"] = old_high
        state["highest_price_since_entry"] = observed_price
        state["current_midpoint_stop"] = max(
            old_stop,
            (old_high + observed_price) / 2,
        )
    else:
        state["highest_price_since_entry"] = old_high
        state["current_midpoint_stop"] = old_stop
    return state


def is_alpaca_unauthorized(error):
    message = str(error).lower()
    return "401" in message or "unauthorized" in message or "not authorized" in message


def raise_if_alpaca_unauthorized(error, action):
    if is_alpaca_unauthorized(error):
        environment = "paper" if ALPACA_PAPER else "live"
        raise AlpacaAuthError(
            f"Alpaca rejected credentials while trying to {action}. "
            f"Check that API_KEY and SECRET_KEY are valid {environment} keys and "
            "that ALPACA_PAPER matches the key type."
        ) from error


def is_transient_alpaca_error(error):
    return isinstance(error, (RequestsConnectionError, Timeout))


def alpaca_read(call, action, retries=ALPACA_READ_RETRIES):
    """Run an idempotent Alpaca read with bounded connection retries."""
    for attempt in range(1, retries + 1):
        try:
            return call()
        except Exception as error:
            raise_if_alpaca_unauthorized(error, action)
            if not is_transient_alpaca_error(error) or attempt == retries:
                raise

            delay = ALPACA_RETRY_DELAY_SECONDS * attempt
            print(
                f"Alpaca connection error while trying to {action} "
                f"(attempt {attempt}/{retries}). Retrying in {delay}s: {error}"
            )
            time.sleep(delay)


def validate_alpaca_credentials():
    missing = [
        name
        for name, value in (("API_KEY", API_KEY), ("SECRET_KEY", SECRET_KEY))
        if not value
    ]
    if missing:
        raise AlpacaAuthError(
            "Missing Alpaca credential environment variables: "
            + ", ".join(missing)
            + ". Add them to .env before starting the bot."
        )

    try:
        alpaca_read(trading_client.get_account, "validate the trading account")
    except Exception as e:
        raise_if_alpaca_unauthorized(e, "validate the trading account")
        raise

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

def log_trade(
    symbol, side, qty, price, reason, entry_price=None, exit_price=None,
    pnl=None, order_id=None, timestamp=None,
):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "environment",
                "order_id",
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
            timestamp or datetime.now(),
            TRADING_ENVIRONMENT,
            order_id,
            symbol,
            side,
            qty,
            price,
            reason,
            entry_price,
            exit_price,
            pnl
        ])


def wait_for_order_fill(order_id):
    """Return Alpaca's final filled order, or None after a bounded wait."""
    for attempt in range(ORDER_FILL_ATTEMPTS):
        order = alpaca_read(
            lambda: trading_client.get_order_by_id(order_id),
            f"check order {order_id} fill status",
        )
        status = str(getattr(order, "status", "")).lower()
        if status.endswith("filled"):
            return order
        if any(status.endswith(value) for value in ("canceled", "expired", "rejected")):
            return None
        if attempt + 1 < ORDER_FILL_ATTEMPTS:
            time.sleep(ORDER_FILL_DELAY_SECONDS)
    return None
        
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=ALPACA_PAPER)

def get_managed_positions():
    """Return only account positions explicitly opened and owned by this bot."""
    positions = alpaca_read(
        trading_client.get_all_positions, "list ETFEnhancer-managed positions"
    )
    return [
        position for position in positions
        if position.symbol in position_state
        and position.symbol in LIVE_TRADING_UNIVERSE
    ]


def managed_entry_price(symbol, position):
    return float(position_state.get(symbol, {}).get("entry_price", position.avg_entry_price))


def managed_quantity(symbol, position):
    return float(position_state.get(symbol, {}).get("qty", position.qty))


def get_total_market_value():
    total = 0.0
    for position in get_managed_positions():
        owned_qty = float(position_state[position.symbol].get("qty", position.qty))
        total += abs(owned_qty) * float(position.current_price)
    return total


def get_managed_performance():
    """Return post-cutoff unrealized performance for ETFEnhancer-owned shares."""
    entry_times_by_order_id = {}
    try:
        with open(LOG_FILE, newline="") as file:
            for row in csv.DictReader(file):
                if row.get("side") == "buy" and row.get("order_id"):
                    entry_times_by_order_id[row["order_id"]] = row.get("timestamp")
    except FileNotFoundError:
        pass

    performance_start = date.fromisoformat(PERFORMANCE_START_DATE)
    cost_basis = 0.0
    market_value = 0.0
    for position in get_managed_positions():
        symbol = position.symbol
        state = position_state[symbol]
        entry_time = state.get("entry_filled_at") or entry_times_by_order_id.get(
            str(state.get("entry_order_id", ""))
        )
        try:
            entry_date = datetime.fromisoformat(str(entry_time)).date()
        except (TypeError, ValueError):
            # Unknown legacy entries are excluded rather than risk including
            # shared-account activity from before the isolation fix.
            continue
        if entry_date < performance_start:
            continue

        qty = abs(managed_quantity(symbol, position))
        cost_basis += qty * managed_entry_price(symbol, position)
        market_value += qty * float(position.current_price)

    unrealized_pnl = market_value - cost_basis
    return_percent = (
        (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
    )
    return cost_basis, market_value, unrealized_pnl, return_percent


def get_open_position_symbols():
    return [position.symbol for position in get_managed_positions()]

from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

def has_open_order(symbol):
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = alpaca_read(
            lambda: trading_client.get_orders(filter=request), "check open orders"
        )

        for order in orders:
            if order.symbol == symbol:
                return True

    except Exception as e:
        raise_if_alpaca_unauthorized(e, "check open orders")
        if is_transient_alpaca_error(e):
            raise
        print(f"Error checking open orders: {e}")

    return False

def already_holding(symbol):
    # Alpaca's position endpoint can briefly lag a confirmed fractional fill.
    # The durable bot state is therefore also an entry guard.
    return symbol in position_state or get_position(symbol) > 0


def reconcile_position_state():
    """Remove state for positions that were closed outside ETFEnhancer.

    Unknown shared-account positions are deliberately not adopted. A pending
    order protects state from being removed while Alpaca is settling a fill.
    """
    positions = alpaca_read(
        trading_client.get_all_positions, "reconcile ETFEnhancer positions"
    )
    actual_symbols = {position.symbol for position in positions}
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = alpaca_read(
        lambda: trading_client.get_orders(filter=request),
        "reconcile ETFEnhancer open orders",
    )
    pending_symbols = {order.symbol for order in orders}
    disallowed_symbols = sorted(
        symbol for symbol in position_state if symbol not in LIVE_TRADING_UNIVERSE
    )
    stale_symbols = sorted(
        symbol for symbol in position_state
        if symbol in LIVE_TRADING_UNIVERSE
        and symbol not in actual_symbols
        and symbol not in pending_symbols
    )
    removed_symbols = disallowed_symbols + stale_symbols
    if removed_symbols:
        for symbol in removed_symbols:
            position_state.pop(symbol, None)
        save_position_state()
        if disallowed_symbols:
            print(
                "Removed non-universe ETFEnhancer position state: "
                + ", ".join(disallowed_symbols)
            )
        if stale_symbols:
            print(
                "Removed stale ETFEnhancer position state: "
                + ", ".join(stale_symbols)
            )
    return removed_symbols

def get_open_positions_count():
    return len(get_managed_positions())

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

        entry_price = managed_entry_price(symbol, position)
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

            qty = managed_quantity(symbol, position)
            if place_trade(symbol, "sell", qty=qty, reason="atr_trailing_stop"):
                mark_recently_sold(symbol)
                highest_price.pop(symbol, None)
                return True

    except Exception as e:
        raise_if_alpaca_unauthorized(e, f"check ATR trailing stop for {symbol}")
        return False

    return False


def check_dynamic_midpoint_stop(symbol):
    """Raise a persisted midpoint stop and exit on an hourly close below it."""
    if has_open_order(symbol):
        print(f"Sell order already pending for {symbol}")
        return True

    try:
        position = trading_client.get_open_position(symbol)
        entry_price = managed_entry_price(symbol, position)
        qty = managed_quantity(symbol, position)
        data = fetch_price_history(symbol, period="1mo", interval="1h")
        if data.empty:
            print(f"Close unavailable for {symbol}. Skipping midpoint stop.")
            return False
        if len(data) < 2:
            return False
        # During market hours Yahoo's last hourly row can still be forming.
        # Evaluate exits only from the preceding, completed bar.
        completed_bar = data.iloc[-2]
        current_close = float(completed_bar["Close"])
        close_timestamp = str(data.index[-2])
        current_price = float(position.current_price)

        state = position_state.setdefault(
            symbol,
            {
                "entry_price": entry_price,
                "previous_high": entry_price,
                "highest_price_since_entry": entry_price,
                "current_midpoint_stop": entry_price,
                "last_evaluated_close": close_timestamp,
            },
        )
        # Alpaca remains authoritative for the average fill price.
        update_midpoint_state(state, entry_price, current_price)
        save_position_state()

        print(f"{symbol} entry: {entry_price}")
        print(f"{symbol} highest price: {state['highest_price_since_entry']}")
        print(f"{symbol} midpoint stop: {state['current_midpoint_stop']}")

        previous_close_timestamp = state.get("last_evaluated_close")
        is_new_close = previous_close_timestamp is not None and previous_close_timestamp != close_timestamp
        state["last_evaluated_close"] = close_timestamp
        save_position_state()

        if is_new_close and current_close < state["current_midpoint_stop"]:
            if place_trade(symbol, "sell", qty=qty, reason="dynamic_midpoint_stop"):
                mark_recently_sold(symbol)
                position_state.pop(symbol, None)
                save_position_state()
                return True
    except Exception as e:
        raise_if_alpaca_unauthorized(e, f"check dynamic midpoint stop for {symbol}")
        return False
    return False


def get_symbol_pivot_state(symbol):
    """Update a symbol from completed weekly closes and persist its ZigZag state."""
    data = fetch_price_history(symbol, period="1y", interval="1d")
    if data.empty:
        return pivot_state.get(symbol)
    clock = alpaca_read(trading_client.get_clock, "check weekly-bar completion")
    cutoff = pd.Timestamp.now().normalize()
    if clock.is_open:
        cutoff -= pd.Timedelta(days=1)
    weekly = completed_weekly_closes(
        data, as_of=cutoff, timeframe=PIVOT_TIMEFRAME,
        price_source=PIVOT_PRICE_SOURCE,
    )
    state = pivot_state.get(symbol)
    signature = {
        "reversal_percent": PIVOT_REVERSAL_PERCENT,
        "lookback_weeks": PIVOT_LOOKBACK_WEEKS,
        "minimum_weeks": MIN_WEEKS_BETWEEN_PIVOTS,
        "timeframe": PIVOT_TIMEFRAME,
        "price_source": PIVOT_PRICE_SOURCE,
    }
    if state is None or state.get("configuration") != signature:
        state, _, _ = build_pivot_history(
            weekly,
            PIVOT_REVERSAL_PERCENT,
            PIVOT_LOOKBACK_WEEKS,
            MIN_WEEKS_BETWEEN_PIVOTS,
        )
        state["configuration"] = signature
        pivot_state[symbol] = state
    else:
        last_week = state.get("last_processed_week")
        for date, close in weekly.items():
            if last_week is not None and date <= pd.Timestamp(last_week):
                continue
            event = update_pivot_state(
                state,
                close,
                date,
                PIVOT_REVERSAL_PERCENT,
                PIVOT_LOOKBACK_WEEKS,
                MIN_WEEKS_BETWEEN_PIVOTS,
            )
            if event.get("message"):
                print(f"{symbol} {event['message']}")
    save_pivot_state()
    return state


def check_structural_midpoint_stop(symbol):
    if has_open_order(symbol):
        print(f"Sell order already pending for {symbol}")
        return True
    try:
        position = trading_client.get_open_position(symbol)
        entry_price = managed_entry_price(symbol, position)
        qty = managed_quantity(symbol, position)
        pivots = get_symbol_pivot_state(symbol)
        if not pivots:
            return False
        state = position_state.setdefault(
            symbol,
            {
                "entry_price": entry_price,
                "trade_anchor_low": pivots.get("confirmed_swing_low"),
                "active_structural_low": pivots.get("confirmed_swing_low"),
                "current_structural_stop": None,
            },
        )
        if state.get("active_structural_low") is None:
            state["trade_anchor_low"] = pivots.get("confirmed_swing_low")
            state["active_structural_low"] = pivots.get("confirmed_swing_low")
        old_stop = state.get("current_structural_stop")
        raised = update_structural_stop(state, pivots, USE_TENTATIVE_HIGH_FOR_STOP)
        pivots["current_structural_stop"] = state.get("current_structural_stop")
        save_pivot_state()
        if raised:
            print(
                f"{symbol} structural stop raised from "
                f"{old_stop if old_stop is not None else 'unset'} to "
                f"{state['current_structural_stop']:.2f}."
            )
        save_position_state()

        print(
            f"{symbol} pivot_direction={pivots.get('pivot_direction')} "
            f"candidate_low={pivots.get('candidate_swing_low')} "
            f"confirmed_low={pivots.get('confirmed_swing_low')} "
            f"candidate_high={pivots.get('candidate_swing_high')} "
            f"confirmed_high={pivots.get('confirmed_swing_high')} "
            f"weekly_close={pivots.get('current_weekly_close')} "
            f"reversal={float(pivots.get('current_reversal_percent', 0)):.1%} "
            f"active_low={state.get('active_structural_low')} "
            f"structural_stop={state.get('current_structural_stop')}"
        )
        stop = state.get("current_structural_stop")
        if stop is None:
            return False
        data = fetch_price_history(symbol, period="1mo", interval=STRUCTURAL_STOP_EXIT_TIMEFRAME)
        if data.empty:
            return False
        clock = alpaca_read(trading_client.get_clock, "check exit-bar completion")
        last_bar_date = pd.Timestamp(data.index[-1]).date()
        today = pd.Timestamp.now().date()
        row_index = -2 if clock.is_open and last_bar_date >= today and len(data) > 1 else -1
        exit_price = float(data["Close"].iloc[row_index])
        breached = exit_price < float(stop)
        if not REQUIRE_CLOSE_BELOW_STRUCTURAL_STOP:
            breached = float(position.current_price) < float(stop)
            exit_price = float(position.current_price)
        if breached:
            print(f"{symbol} exiting: completed close {exit_price:.2f} below structural stop {stop:.2f}.")
            if place_trade(symbol, "sell", qty=qty, reason="structural_midpoint_stop"):
                mark_recently_sold(symbol)
                position_state.pop(symbol, None)
                save_position_state()
                return True
    except Exception as e:
        raise_if_alpaca_unauthorized(e, f"check structural midpoint stop for {symbol}")
        print(f"Error checking structural midpoint stop for {symbol}: {e}")
    return False


def check_trailing_stop(symbol, trailing_percent):
    print("Fixed percent trailing stops are deprecated; using ATR trailing stop.")
    return check_atr_trailing_stop(symbol, ATR_TRAILING_MULTIPLIER)

def get_position(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except APIError as e:
        raise_if_alpaca_unauthorized(e, f"get the {symbol} position")
        return 0

def check_stop_loss(symbol, stop_loss_percent):
    if has_open_order(symbol):
        print(f"Sell order already pending for {symbol}")
        return True
    
    try:
        position = trading_client.get_open_position(symbol)

        entry_price = managed_entry_price(symbol, position)
        current_price = float(position.current_price)

        loss_threshold = entry_price * (1 - stop_loss_percent)

        print(f"Entry Price: {entry_price}")
        print(f"Current Price: {current_price}")
        print(f"Stop Loss Price: {loss_threshold}")

        if current_price <= loss_threshold:
            print("STOP LOSS TRIGGERED!")

            qty = managed_quantity(symbol, position)
            if place_trade(symbol, "sell", qty=qty, reason="stop_loss"):
                mark_recently_sold(symbol)
                return True

    except Exception as e:
        raise_if_alpaca_unauthorized(e, f"check stop loss for {symbol}")
        return False

    return False

def place_trade(
    symbol, side, qty=None, notional=15, reason="signal", position_role="tactical"
):
    symbol = symbol.strip().upper() if isinstance(symbol, str) else symbol
    if not is_live_trading_symbol(symbol):
        print(
            f"BLOCKED ORDER: {symbol!r} is outside ETFEnhancer's live ETF universe."
        )
        return False

    if side == "buy" and symbol in position_state:
        print("Already holding position. Skipping buy.")
        return False

    current_position = get_position(symbol)
    entry_price = None
    if side == "sell":
        managed_state = position_state.get(symbol)
        if managed_state is None:
            print(f"{symbol} is not owned by ETFEnhancer. Skipping sell.")
            return False
        entry_price = float(managed_state["entry_price"])
        owned_qty = float(managed_state.get("qty", qty or current_position))
        qty = min(float(qty or owned_qty), owned_qty)

    if has_open_order(symbol):
        print(f"Open order exists for {symbol}. Skipping...")
        return False

    if side == "buy" and current_position > 0:
        print("Already holding position. Skipping buy.")
        return False

    if side == "sell" and current_position == 0:
        print("No shares to sell. Skipping sell.")
        return False

    if side == "buy":
        order = MarketOrderRequest(
            symbol=symbol,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"{ORDER_CLIENT_ID_PREFIX}-{uuid.uuid4().hex[:20]}",
        )

    else:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"{ORDER_CLIENT_ID_PREFIX}-{uuid.uuid4().hex[:20]}",
        )

    try:
        submitted_order = trading_client.submit_order(order)
        filled_order = wait_for_order_fill(submitted_order.id)
        if filled_order is None:
            print(
                f"Order {submitted_order.id} for {symbol} was not confirmed filled; "
                "it was not added to the execution log."
            )
            return False

        filled_price = float(filled_order.filled_avg_price)
        filled_qty = float(filled_order.filled_qty)
        filled_at = getattr(filled_order, "filled_at", None)
        order_id = str(filled_order.id)

        if side == "buy":
            print(f"Filled BUY order for {filled_qty} shares of {symbol} at ${filled_price}")
            entry_price = filled_price
            log_trade(
                symbol=symbol,
                side="buy",
                qty=filled_qty,
                price=filled_price,
                reason=reason,
                entry_price=entry_price,
                exit_price=None,
                pnl=None,
                order_id=order_id,
                timestamp=filled_at,
            )
            previous = position_state.get(symbol, {})
            previous_qty = float(previous.get("qty", 0.0))
            combined_qty = previous_qty + filled_qty
            combined_entry = (
                float(previous.get("entry_price", entry_price)) * previous_qty
                + entry_price * filled_qty
            ) / combined_qty
            position_state[symbol] = {
                "entry_price": combined_entry,
                "qty": combined_qty,
                "entry_order_id": order_id,
                "entry_filled_at": (
                    filled_at.isoformat()
                    if hasattr(filled_at, "isoformat")
                    else str(filled_at or datetime.now().isoformat())
                ),
                "position_role": previous.get("position_role", position_role),
                "trade_anchor_low": previous.get("trade_anchor_low"),
                "active_structural_low": previous.get("active_structural_low"),
                "current_structural_stop": previous.get("current_structural_stop"),
            }
            pivots = get_symbol_pivot_state(symbol)
            if pivots:
                anchor = pivots.get("confirmed_swing_low")
                position_state[symbol]["trade_anchor_low"] = anchor
                position_state[symbol]["active_structural_low"] = anchor
            save_position_state()

        else:
            pnl = (filled_price - entry_price) * filled_qty
            print(f"Filled SELL order for {filled_qty} shares of {symbol} at ${filled_price}")
            log_trade(
                symbol=symbol,
                side="sell",
                qty=filled_qty,
                price=filled_price,
                reason=reason,
                entry_price=entry_price,
                exit_price=filled_price,
                pnl=pnl,
                order_id=order_id,
                timestamp=filled_at,
            )
            position_state.pop(symbol, None)
            save_position_state()
        return True

    except Exception as e:
        print(f"Order failed: {e}")
        return False

def print_account_info():
    account = alpaca_read(trading_client.get_account, "get account information")
    _, managed_value, managed_pnl, managed_return_percent = get_managed_performance()

    print("\n===== ACCOUNT INFO =====")
    print(f"Shared Account Equity: ${account.equity}")
    print(f"Shared Account Cash: ${account.cash}")
    print(f"ETFEnhancer Position Value: ${managed_value:.2f}")
    print(
        f"ETFEnhancer Unrealized Gain/Loss Since {PERFORMANCE_START_DATE}: "
        f"${managed_pnl:+.2f} ({managed_return_percent:+.2f}%)"
    )
    print(f"Shared Account Buying Power: ${account.buying_power}")
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
