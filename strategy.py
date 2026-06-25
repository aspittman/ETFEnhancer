import time

import pandas as pd
import ta
import yfinance as yf


def wait_for_market_open(trading_client):
    print("Checking if market is open...")

    while True:
        clock = trading_client.get_clock()

        if clock.is_open:
            print("Market is OPEN! Starting strategy...")
            break
        else:
            print("Market closed. Waiting 60 seconds...")
            time.sleep(60)


def fetch_price_history(symbol, period="1y", interval="1h"):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    return normalize_price_data(data)


def normalize_price_data(data):
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        return pd.DataFrame()

    clean = data[required].copy()
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    return clean.dropna()


def build_strategy_frame(
    data,
    ma_short,
    ma_long,
    macd_fast,
    macd_slow,
    macd_signal,
    atr_window=14,
):
    clean = normalize_price_data(data)
    if clean.empty:
        return clean

    frame = clean.copy()
    close = frame["Close"]

    frame["ma_short"] = close.rolling(window=ma_short).mean()
    frame["ma_long"] = close.rolling(window=ma_long).mean()
    frame["prev_ma_short"] = frame["ma_short"].shift(1)

    macd = ta.trend.MACD(
        close=close,
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    frame["macd"] = macd.macd()
    frame["macd_signal"] = macd.macd_signal()
    frame["macd_hist"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(
        high=frame["High"],
        low=frame["Low"],
        close=close,
        window=atr_window,
    )
    frame["atr"] = atr.average_true_range()
    frame["atr_percent"] = frame["atr"] / close * 100

    frame["in_uptrend"] = (close > frame["ma_short"]) & (
        frame["ma_short"] > frame["ma_long"]
    )
    frame["ma_rising"] = frame["ma_short"] > frame["prev_ma_short"]
    frame["macd_confirmed"] = (frame["macd"] > frame["macd_signal"]) & (
        frame["macd_hist"] > 0
    )
    frame["signal"] = (
        frame["in_uptrend"] & frame["ma_rising"] & frame["macd_confirmed"]
    )

    trend_score = (close - frame["ma_short"]) / frame["ma_short"] * 100
    structure_score = (
        (frame["ma_short"] - frame["ma_long"]) / frame["ma_long"] * 100
    )
    momentum_score = frame["macd_hist"]
    volatility_penalty = frame["atr_percent"].clip(lower=0) * 0.15
    frame["score"] = trend_score + structure_score + momentum_score - volatility_penalty

    return frame


def signal_from_row(symbol, row):
    values = [
        row.get("Close"),
        row.get("ma_short"),
        row.get("ma_long"),
        row.get("prev_ma_short"),
        row.get("macd"),
        row.get("macd_signal"),
        row.get("macd_hist"),
        row.get("score"),
        row.get("atr"),
    ]
    if any(pd.isna(value) for value in values):
        return None

    if not bool(row.get("signal")):
        return None

    return {
        "symbol": symbol,
        "price": float(row["Close"]),
        "score": float(row["score"]),
        "ma_short": float(row["ma_short"]),
        "ma_long": float(row["ma_long"]),
        "macd": float(row["macd"]),
        "macd_signal": float(row["macd_signal"]),
        "macd_hist": float(row["macd_hist"]),
        "atr": float(row["atr"]),
        "atr_percent": float(row["atr_percent"]),
    }


def check_signal(
    symbol,
    ma_short,
    ma_long,
    macd_fast,
    macd_slow,
    macd_signal,
    atr_window=14,
    period="1y",
    interval="1h",
):
    try:
        data = fetch_price_history(symbol, period=period, interval=interval)

        if data.empty or len(data) < ma_long + 5:
            return None

        frame = build_strategy_frame(
            data,
            ma_short,
            ma_long,
            macd_fast,
            macd_slow,
            macd_signal,
            atr_window,
        )
        if frame.empty:
            return None

        return signal_from_row(symbol, frame.iloc[-1])

    except Exception as e:
        print(f"Error checking {symbol}: {e}")
        return None


def get_market_regime(
    symbol,
    ma_short=50,
    ma_long=200,
    period="1y",
    interval="1d",
):
    try:
        data = fetch_price_history(symbol, period=period, interval=interval)
        if data.empty or len(data) < ma_long + 5:
            return {
                "symbol": symbol,
                "is_healthy": False,
                "reason": "not_enough_data",
            }

        close = data["Close"]
        short_ma = close.rolling(window=ma_short).mean()
        long_ma = close.rolling(window=ma_long).mean()

        latest_close = float(close.iloc[-1])
        latest_short = float(short_ma.iloc[-1])
        latest_long = float(long_ma.iloc[-1])
        previous_short = float(short_ma.iloc[-2])

        values = [latest_close, latest_short, latest_long, previous_short]
        if any(pd.isna(value) for value in values):
            return {"symbol": symbol, "is_healthy": False, "reason": "nan_values"}

        is_healthy = (
            latest_close > latest_short
            and latest_short > latest_long
            and latest_short >= previous_short
        )

        return {
            "symbol": symbol,
            "is_healthy": is_healthy,
            "reason": "healthy" if is_healthy else "weak_market",
            "price": latest_close,
            "ma_short": latest_short,
            "ma_long": latest_long,
        }

    except Exception as e:
        print(f"Error checking market regime for {symbol}: {e}")
        return {"symbol": symbol, "is_healthy": False, "reason": "error"}


def scan_universe(
    universe,
    ma_short,
    ma_long,
    macd_fast,
    macd_slow,
    macd_signal,
    atr_window=14,
    min_score=None,
    blocked_symbols=None,
):
    candidates = []
    blocked = set(blocked_symbols or [])

    for symbol in universe:
        if symbol in blocked:
            print(f"{symbol} is blocked from new entries. Skipping scan.")
            continue

        print(f"Scanning {symbol}...")
        result = check_signal(
            symbol,
            ma_short,
            ma_long,
            macd_fast,
            macd_slow,
            macd_signal,
            atr_window,
        )

        if result and (min_score is None or result["score"] >= min_score):
            candidates.append(result)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
