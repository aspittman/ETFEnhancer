import argparse
from dataclasses import dataclass

import pandas as pd

from analytics import print_summary, summarize_closed_trades
from config import (
    ATR_TRAILING_MULTIPLIER,
    ATR_WINDOW,
    BLOCKED_SYMBOLS,
    DOLLARS_PER_TRADE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MARKET_REGIME_SYMBOL,
    MAX_BUYS_PER_CYCLE,
    MAX_POSITIONS,
    MA_LONG,
    MA_SHORT,
    MIN_CANDIDATE_SCORE,
    STOP_LOSS_PERCENT,
)
from strategy import build_strategy_frame, fetch_price_history, signal_from_row
from universe import UNIVERSE


@dataclass
class BacktestConfig:
    period: str = "1y"
    interval: str = "1h"
    dollars_per_trade: float = DOLLARS_PER_TRADE
    max_positions: int = MAX_POSITIONS
    max_buys_per_bar: int = MAX_BUYS_PER_CYCLE
    stop_loss_percent: float = STOP_LOSS_PERCENT
    atr_multiplier: float = ATR_TRAILING_MULTIPLIER
    atr_window: int = ATR_WINDOW
    min_candidate_score: float = MIN_CANDIDATE_SCORE


def run_backtest(symbols=None, config=None, blocked_symbols=None):
    config = config or BacktestConfig()
    symbols = list(symbols or UNIVERSE)
    blocked = set(blocked_symbols if blocked_symbols is not None else BLOCKED_SYMBOLS)

    frames = {}
    for symbol in symbols:
        if symbol in blocked:
            continue

        print(f"Loading {symbol}...")
        data = fetch_price_history(symbol, period=config.period, interval=config.interval)
        frame = build_strategy_frame(
            data,
            MA_SHORT,
            MA_LONG,
            MACD_FAST,
            MACD_SLOW,
            MACD_SIGNAL,
            config.atr_window,
        )
        if not frame.empty:
            frames[symbol] = frame

    if not frames:
        return {"trades": [], "summary": summarize_closed_trades([])}

    benchmark = frames.get(MARKET_REGIME_SYMBOL)
    if benchmark is None:
        data = fetch_price_history(
            MARKET_REGIME_SYMBOL,
            period=config.period,
            interval=config.interval,
        )
        benchmark = build_strategy_frame(
            data,
            MA_SHORT,
            MA_LONG,
            MACD_FAST,
            MACD_SLOW,
            MACD_SIGNAL,
            config.atr_window,
        )

    calendar = sorted(set().union(*(frame.index for frame in frames.values())))
    positions = {}
    closed_trades = []

    for timestamp in calendar:
        _manage_positions(timestamp, frames, positions, closed_trades, config)

        if not _market_is_healthy(timestamp, benchmark):
            continue

        candidates = _rank_candidates(timestamp, frames, positions, config)
        buys = 0
        for candidate in candidates:
            if len(positions) >= config.max_positions:
                break
            if buys >= config.max_buys_per_bar:
                break

            symbol = candidate["symbol"]
            price = candidate["price"]
            qty = config.dollars_per_trade / price
            positions[symbol] = {
                "symbol": symbol,
                "entry_time": timestamp,
                "entry_price": price,
                "qty": qty,
                "highest_price": price,
                "entry_score": candidate["score"],
            }
            buys += 1

    final_timestamp = calendar[-1]
    for symbol in list(positions):
        frame = frames.get(symbol)
        if frame is None or final_timestamp not in frame.index:
            continue
        row = frame.loc[final_timestamp]
        _close_position(
            symbol,
            positions,
            closed_trades,
            final_timestamp,
            float(row["Close"]),
            "end_of_backtest",
        )

    return {
        "trades": closed_trades,
        "summary": summarize_closed_trades(closed_trades),
    }


def _manage_positions(timestamp, frames, positions, closed_trades, config):
    for symbol in list(positions):
        frame = frames.get(symbol)
        if frame is None or timestamp not in frame.index:
            continue

        row = frame.loc[timestamp]
        if pd.isna(row.get("Close")) or pd.isna(row.get("Low")) or pd.isna(row.get("High")):
            continue

        position = positions[symbol]
        position["highest_price"] = max(position["highest_price"], float(row["High"]))

        stop_price = position["entry_price"] * (1 - config.stop_loss_percent)
        atr = row.get("atr")
        trail_price = None
        if not pd.isna(atr) and float(atr) > 0:
            trail_price = position["highest_price"] - (float(atr) * config.atr_multiplier)

        low = float(row["Low"])
        if low <= stop_price:
            _close_position(
                symbol,
                positions,
                closed_trades,
                timestamp,
                stop_price,
                "stop_loss",
            )
        elif trail_price is not None and low <= trail_price:
            _close_position(
                symbol,
                positions,
                closed_trades,
                timestamp,
                trail_price,
                "atr_trailing_stop",
            )


def _rank_candidates(timestamp, frames, positions, config):
    candidates = []
    for symbol, frame in frames.items():
        if symbol in positions or timestamp not in frame.index:
            continue

        signal = signal_from_row(symbol, frame.loc[timestamp])
        if signal and signal["score"] >= config.min_candidate_score:
            candidates.append(signal)

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def _market_is_healthy(timestamp, benchmark):
    if benchmark is None or benchmark.empty:
        return False

    available = benchmark.index[benchmark.index <= timestamp]
    if len(available) == 0:
        return False

    row = benchmark.loc[available[-1]]
    values = [row.get("Close"), row.get("ma_short"), row.get("ma_long"), row.get("prev_ma_short")]
    if any(pd.isna(value) for value in values):
        return False

    return (
        float(row["Close"]) > float(row["ma_short"])
        and float(row["ma_short"]) > float(row["ma_long"])
        and float(row["ma_short"]) >= float(row["prev_ma_short"])
    )


def _close_position(symbol, positions, closed_trades, timestamp, exit_price, reason):
    position = positions.pop(symbol)
    pnl = (exit_price - position["entry_price"]) * position["qty"]
    closed_trades.append(
        {
            "symbol": symbol,
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "qty": position["qty"],
            "pnl": pnl,
            "return_pct": (exit_price - position["entry_price"]) / position["entry_price"],
            "entry_score": position["entry_score"],
            "exit_reason": reason,
        }
    )


def _parse_symbols(value):
    if not value:
        return UNIVERSE
    return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an offline strategy backtest.")
    parser.add_argument("--symbols", help="Comma-separated symbols. Defaults to universe.py.")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--dollars-per-trade", type=float, default=DOLLARS_PER_TRADE)
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument("--max-buys-per-bar", type=int, default=MAX_BUYS_PER_CYCLE)
    parser.add_argument("--min-score", type=float, default=MIN_CANDIDATE_SCORE)
    args = parser.parse_args()

    result = run_backtest(
        symbols=_parse_symbols(args.symbols),
        config=BacktestConfig(
            period=args.period,
            interval=args.interval,
            dollars_per_trade=args.dollars_per_trade,
            max_positions=args.max_positions,
            max_buys_per_bar=args.max_buys_per_bar,
            min_candidate_score=args.min_score,
        ),
    )
    print_summary(result["summary"])
