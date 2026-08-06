import csv
from collections import defaultdict
from datetime import datetime
from collections import Counter
from statistics import median

import pandas as pd
import yfinance as yf


LEGACY_TRADE_LOG_FIELDS = [
    "timestamp",
    "symbol",
    "side",
    "qty",
    "price",
    "reason",
    "entry_price",
    "exit_price",
    "pnl",
]

TRADE_LOG_FIELDS = [
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
    "pnl",
]


def summarize_closed_trades(trades):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "average_holding_period_hours": 0.0,
            "median_holding_period_hours": 0.0,
            "total_return": 0.0,
            "exits_by_reason": {},
            "confirmed_pivots": 0,
            "average_pivot_confirmation_weeks": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "best_symbol": None,
            "worst_symbol": None,
            "by_symbol": {},
        }

    df = pd.DataFrame(trades)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum())
    gross_loss = abs(float(losses["pnl"].sum()))

    by_symbol = {}
    for symbol, group in df.groupby("symbol"):
        symbol_wins = group[group["pnl"] > 0]
        by_symbol[symbol] = {
            "trades": int(len(group)),
            "win_rate": float(len(symbol_wins) / len(group)),
            "total_pnl": float(group["pnl"].sum()),
            "expectancy": float(group["pnl"].mean()),
        }

    ranked_symbols = sorted(
        by_symbol.items(),
        key=lambda item: item[1]["total_pnl"],
        reverse=True,
    )
    ordered = df.sort_values("exit_time") if "exit_time" in df.columns else df
    equity = ordered["pnl"].cumsum()
    drawdown = equity - equity.cummax()
    holding_hours = [_holding_hours(row) for row in trades]
    holding_hours = [hours for hours in holding_hours if hours is not None]
    deployed_capital = sum(
        (_to_float(trade.get("entry_price")) or 0.0)
        * (_to_float(trade.get("qty")) or 0.0)
        for trade in trades
    )

    return {
        "total_trades": int(len(df)),
        "win_rate": float(len(wins) / len(df)),
        "expectancy": float(df["pnl"].mean()),
        "total_pnl": float(df["pnl"].sum()),
        "average_win": float(wins["pnl"].mean()) if not wins.empty else 0.0,
        "average_loss": float(losses["pnl"].mean()) if not losses.empty else 0.0,
        "average_holding_period_hours": sum(holding_hours) / len(holding_hours) if holding_hours else 0.0,
        "median_holding_period_hours": median(holding_hours) if holding_hours else 0.0,
        "total_return": float(df["pnl"].sum()) / deployed_capital if deployed_capital else 0.0,
        "exits_by_reason": dict(Counter(str(trade.get("exit_reason", "unknown")) for trade in trades)),
        "confirmed_pivots": 0,
        "average_pivot_confirmation_weeks": 0.0,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        ),
        "max_drawdown": abs(float(drawdown.min())) if not drawdown.empty else 0.0,
        "best_symbol": ranked_symbols[0][0] if ranked_symbols else None,
        "worst_symbol": ranked_symbols[-1][0] if ranked_symbols else None,
        "by_symbol": by_symbol,
    }


def add_benchmark_comparison(summary, trades, benchmark_closes, symbol="SPY"):
    """Compare each strategy trade with a benchmark over the same timestamps."""
    result = dict(summary)
    closes = _normalize_benchmark_closes(benchmark_closes)
    comparisons = []

    for trade in trades:
        entry_price = _to_float(trade.get("entry_price"))
        exit_price = _to_float(trade.get("exit_price"))
        qty = _to_float(trade.get("qty")) or 0.0
        entry_time = _to_timestamp(trade.get("entry_time"))
        exit_time = _to_timestamp(trade.get("exit_time"))
        if None in (entry_price, exit_price, entry_time, exit_time):
            continue

        benchmark_entry = _price_at_or_before(closes, entry_time)
        benchmark_exit = _price_at_or_before(closes, exit_time)
        if benchmark_entry is None or benchmark_exit is None or benchmark_entry <= 0:
            continue

        strategy_return = (exit_price - entry_price) / entry_price
        benchmark_return = (benchmark_exit - benchmark_entry) / benchmark_entry
        comparisons.append(
            {
                "capital": entry_price * qty,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
            }
        )

    result["benchmark_symbol"] = symbol
    result["benchmark_trade_count"] = len(comparisons)
    if not comparisons:
        return result

    total_weight = sum(item["capital"] for item in comparisons)
    if total_weight <= 0:
        total_weight = float(len(comparisons))
        for item in comparisons:
            item["capital"] = 1.0

    def weighted(field):
        return sum(item[field] * item["capital"] for item in comparisons) / total_weight

    first_entry = min(item["entry_time"] for item in comparisons)
    last_exit = max(item["exit_time"] for item in comparisons)
    period_start = _price_at_or_before(closes, first_entry)
    period_end = _price_at_or_before(closes, last_exit)

    result.update(
        {
            "matched_strategy_return": weighted("strategy_return"),
            "matched_benchmark_return": weighted("benchmark_return"),
            "matched_excess_return": weighted("excess_return"),
            "benchmark_outperformance_rate": sum(
                item["excess_return"] > 0 for item in comparisons
            ) / len(comparisons),
            "benchmark_buy_hold_return": (
                (period_end - period_start) / period_start
                if period_start and period_end is not None
                else None
            ),
        }
    )
    return result


def fetch_benchmark_closes(trades, symbol="SPY"):
    timestamps = [
        timestamp
        for trade in trades
        for timestamp in (
            _to_timestamp(trade.get("entry_time")),
            _to_timestamp(trade.get("exit_time")),
        )
        if timestamp is not None
    ]
    if not timestamps:
        return pd.Series(dtype=float)

    start = min(timestamps) - pd.Timedelta(days=7)
    end = max(timestamps) + pd.Timedelta(days=2)
    data = yf.download(
        symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1h",
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty or "Close" not in data:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


def load_live_trade_log(path):
    with open(path, newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        return []

    # Older logs were created without a header when the file already existed.
    # Detect that format instead of allowing DictReader to consume the first
    # trade as field names.
    first_row = [value.strip().lower() for value in rows[0]]
    has_header = "timestamp" in first_row and "symbol" in first_row and "side" in first_row
    data_rows = rows[1:] if has_header else rows

    if has_header:
        fields = [value.strip() for value in rows[0]]
    else:
        fields = LEGACY_TRADE_LOG_FIELDS

    return [dict(zip(fields, row + [""] * (len(fields) - len(row)))) for row in data_rows if row]


def pair_live_trade_log(rows):
    open_entries = defaultdict(list)
    closed_trades = []

    for row in rows:
        symbol = row.get("symbol")
        side = row.get("side")
        if not symbol or side not in {"buy", "sell"}:
            continue

        environment = row.get("environment") or "legacy"
        position_key = (environment, symbol)

        if side == "buy":
            open_entries[position_key].append(row)
            continue

        if not open_entries[position_key]:
            continue

        entry = open_entries[position_key].pop(0)
        entry_price = _to_float(row.get("entry_price")) or _to_float(entry.get("price"))
        exit_price = _to_float(row.get("exit_price")) or _to_float(row.get("price"))
        qty = _to_float(row.get("qty")) or _to_float(entry.get("qty")) or 0.0
        pnl = _to_float(row.get("pnl"))
        if pnl is None and entry_price is not None and exit_price is not None:
            pnl = (exit_price - entry_price) * qty

        closed_trades.append(
            {
                "symbol": symbol,
                "environment": environment,
                "entry_time": entry.get("timestamp"),
                "exit_time": row.get("timestamp"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "pnl": pnl or 0.0,
                "exit_reason": row.get("reason"),
            }
        )

    return closed_trades


def print_summary(summary):
    print("===== TRADE ANALYTICS =====")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Expectancy: ${summary['expectancy']:.2f}")
    print(f"Total P/L: ${summary['total_pnl']:.2f}")
    print(f"Total return: {summary.get('total_return', 0.0):.2%}")
    print(f"Average win: ${summary['average_win']:.2f}")
    print(f"Average loss: ${summary['average_loss']:.2f}")
    print(f"Average holding period: {summary['average_holding_period_hours']:.1f} hours")
    print(f"Median holding period: {summary.get('median_holding_period_hours', 0.0):.1f} hours")
    print(f"Profit factor: {summary['profit_factor']:.2f}")
    print(f"Max drawdown: ${summary.get('max_drawdown', 0.0):.2f}")
    if summary.get("benchmark_trade_count"):
        benchmark = summary["benchmark_symbol"]
        print(f"\n===== {benchmark} BENCHMARK =====")
        print(f"Matched completed trades: {summary['benchmark_trade_count']}")
        print(f"Strategy return on matched trade capital: {summary['matched_strategy_return']:.2%}")
        print(f"{benchmark} return over matched holding windows: {summary['matched_benchmark_return']:.2%}")
        print(f"Excess return versus {benchmark}: {summary['matched_excess_return']:.2%}")
        print(f"Trades outperforming {benchmark}: {summary['benchmark_outperformance_rate']:.2%}")
        if summary.get("benchmark_buy_hold_return") is not None:
            print(f"{benchmark} buy-and-hold over observation period: {summary['benchmark_buy_hold_return']:.2%}")
    print(f"Best symbol: {summary['best_symbol']}")
    print(f"Worst symbol: {summary['worst_symbol']}")
    print(f"Exits by reason: {summary.get('exits_by_reason', {})}")
    print(f"Confirmed pivots: {summary.get('confirmed_pivots', 0)}")
    print(
        "Average pivot confirmation time: "
        f"{summary.get('average_pivot_confirmation_weeks', 0.0):.1f} weeks"
    )
    print("\nBy symbol:")
    for symbol, stats in sorted(
        summary["by_symbol"].items(),
        key=lambda item: item[1]["total_pnl"],
        reverse=True,
    ):
        print(
            f"{symbol}: trades={stats['trades']} "
            f"win_rate={stats['win_rate']:.2%} "
            f"expectancy=${stats['expectancy']:.2f} "
            f"pnl=${stats['total_pnl']:.2f}"
        )


def _to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _holding_hours(trade):
    try:
        start = datetime.fromisoformat(str(trade.get("entry_time")))
        end = datetime.fromisoformat(str(trade.get("exit_time")))
        return (end - start).total_seconds() / 3600
    except (TypeError, ValueError):
        return None


def _to_timestamp(value):
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp
    except (TypeError, ValueError):
        return None


def _normalize_benchmark_closes(values):
    closes = pd.Series(values, copy=True).dropna().astype(float).sort_index()
    index = pd.DatetimeIndex(closes.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    closes.index = index
    return closes[~closes.index.duplicated(keep="last")]


def _price_at_or_before(closes, timestamp):
    if closes.empty or timestamp is None:
        return None
    available = closes.index[closes.index <= timestamp]
    return float(closes.loc[available[-1]]) if len(available) else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a live or paper execution log")
    parser.add_argument("path", nargs="?", default="logs/trades_paper.csv")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    rows = load_live_trade_log(args.path)
    trades = pair_live_trade_log(rows)
    summary = summarize_closed_trades(trades)
    if trades and not args.no_benchmark:
        try:
            closes = fetch_benchmark_closes(trades, args.benchmark)
            summary = add_benchmark_comparison(summary, trades, closes, args.benchmark)
        except Exception as error:
            print(f"Benchmark unavailable: {error}")
    print_summary(summary)
