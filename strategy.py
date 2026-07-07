import time
from dataclasses import dataclass

import pandas as pd
import ta
import yfinance as yf


@dataclass
class StrategySwitches:
    market_regime: bool = True
    ma_alignment: bool = True
    macd: bool = True
    relative_strength: bool = True
    volume: bool = True
    atr_trend_quality: bool = True
    min_score: bool = True
    top_candidates: bool = True


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

    columns = required + (["Volume"] if "Volume" in data.columns else [])
    clean = data[columns].copy()
    for column in columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    return clean.dropna(subset=required)


def build_strategy_frame(
    data,
    ma_short,
    ma_long,
    macd_fast,
    macd_slow,
    macd_signal,
    atr_window=14,
    benchmark_frame=None,
    relative_strength_window=63,
    switches=None,
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
    frame["volume_20"] = (
        frame["Volume"].rolling(window=20).mean() if "Volume" in frame.columns else pd.NA
    )

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

    frame = apply_relative_strength(frame, benchmark_frame, relative_strength_window)
    return score_strategy_frame(frame, switches)


def apply_relative_strength(frame, benchmark_frame=None, window=63):
    frame = frame.copy()
    stock_return = frame["Close"].pct_change(window) * 100
    frame["relative_strength"] = 0.0

    if benchmark_frame is not None and not benchmark_frame.empty:
        benchmark = normalize_price_data(benchmark_frame)
        if not benchmark.empty:
            benchmark_return = benchmark["Close"].pct_change(window) * 100
            benchmark_return = benchmark_return.reindex(frame.index, method="ffill")
            frame["relative_strength"] = stock_return - benchmark_return

    frame["relative_strength"] = pd.to_numeric(
        frame["relative_strength"], errors="coerce"
    ).fillna(0.0)

    return frame


def score_strategy_frame(frame, switches=None):
    switches = switches or StrategySwitches()
    frame = frame.copy()

    frame["ma_alignment_score"] = (
        ((frame["Close"] - frame["ma_short"]) / frame["ma_short"] * 100)
        + ((frame["ma_short"] - frame["ma_long"]) / frame["ma_long"] * 100)
    )
    frame["macd_hist_score"] = frame["macd_hist"] / frame["Close"] * 100
    frame["relative_strength_score"] = frame["relative_strength"]
    if "Volume" in frame.columns:
        frame["volume_score"] = ((frame["Volume"] / frame["volume_20"]) - 1) * 10
    else:
        frame["volume_score"] = 0.0
    frame["atr_trend_quality_score"] = (
        (frame["ma_short"] - frame["ma_long"]) / frame["atr"]
    ).clip(lower=-5, upper=5)

    score = pd.Series(0.0, index=frame.index)
    if switches.ma_alignment:
        score = score + frame["ma_alignment_score"]
    if switches.macd:
        score = score + frame["macd_hist_score"]
    if switches.relative_strength:
        score = score + frame["relative_strength_score"]
    if switches.volume:
        score = score + frame["volume_score"]
    if switches.atr_trend_quality:
        score = score + frame["atr_trend_quality_score"]

    frame["score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return frame


def signal_from_row(symbol, row, switches=None):
    switches = switches or StrategySwitches()
    values = [
        row.get("Close"),
        row.get("ma_short"),
        row.get("ma_long"),
        row.get("prev_ma_short"),
        row.get("score"),
        row.get("atr"),
    ]
    if switches.macd:
        values.extend([row.get("macd"), row.get("macd_signal"), row.get("macd_hist")])
    if switches.relative_strength:
        values.append(row.get("relative_strength"))
    if switches.volume:
        values.append(row.get("volume_score"))
    if switches.atr_trend_quality:
        values.append(row.get("atr_trend_quality_score"))

    if any(pd.isna(value) for value in values):
        return None

    passes_filters = True
    if switches.ma_alignment:
        passes_filters = passes_filters and bool(row.get("in_uptrend")) and bool(
            row.get("ma_rising")
        )
    if switches.macd:
        passes_filters = passes_filters and bool(row.get("macd_confirmed"))
    if not passes_filters:
        return None

    return {
        "symbol": symbol,
        "price": float(row["Close"]),
        "score": float(row["score"]),
        "ma_alignment_score": float(row["ma_alignment_score"]),
        "macd_hist_score": float(row["macd_hist_score"]),
        "relative_strength": float(row["relative_strength"]),
        "relative_strength_score": float(row["relative_strength_score"]),
        "volume_score": float(row["volume_score"]),
        "atr_trend_quality_score": float(row["atr_trend_quality_score"]),
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
    benchmark_frame=None,
    switches=None,
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
            benchmark_frame,
            switches=switches,
        )
        if frame.empty:
            return None

        return signal_from_row(symbol, frame.iloc[-1], switches)

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

        is_healthy = latest_close > latest_long and latest_short > latest_long

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
    max_candidates=None,
    blocked_symbols=None,
    market_regime_symbol="SPY",
    switches=None,
):
    candidates = []
    blocked = set(blocked_symbols or [])
    switches = switches or StrategySwitches()
    benchmark_frame = None
    if switches.relative_strength:
        benchmark_frame = fetch_price_history(
            market_regime_symbol, period="1y", interval="1h"
        )

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
            benchmark_frame=benchmark_frame,
            switches=switches,
        )

        if not result:
            continue

        passes_score = (
            min_score is None or not switches.min_score or result["score"] >= min_score
        )
        if passes_score:
            candidates.append(result)

    if switches.top_candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        if max_candidates is not None:
            return candidates[:max_candidates]
    return candidates
