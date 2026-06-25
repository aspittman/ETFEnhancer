import csv
from collections import defaultdict

import pandas as pd


def summarize_closed_trades(trades):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
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

    return {
        "total_trades": int(len(df)),
        "win_rate": float(len(wins) / len(df)),
        "expectancy": float(df["pnl"].mean()),
        "total_pnl": float(df["pnl"].sum()),
        "average_win": float(wins["pnl"].mean()) if not wins.empty else 0.0,
        "average_loss": float(losses["pnl"].mean()) if not losses.empty else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        "best_symbol": ranked_symbols[0][0] if ranked_symbols else None,
        "worst_symbol": ranked_symbols[-1][0] if ranked_symbols else None,
        "by_symbol": by_symbol,
    }


def load_live_trade_log(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def pair_live_trade_log(rows):
    open_entries = defaultdict(list)
    closed_trades = []

    for row in rows:
        symbol = row.get("symbol")
        side = row.get("side")
        if not symbol or side not in {"buy", "sell"}:
            continue

        if side == "buy":
            open_entries[symbol].append(row)
            continue

        if not open_entries[symbol]:
            continue

        entry = open_entries[symbol].pop(0)
        entry_price = _to_float(row.get("entry_price")) or _to_float(entry.get("price"))
        exit_price = _to_float(row.get("exit_price")) or _to_float(row.get("price"))
        qty = _to_float(row.get("qty")) or _to_float(entry.get("qty")) or 0.0
        pnl = _to_float(row.get("pnl"))
        if pnl is None and entry_price is not None and exit_price is not None:
            pnl = (exit_price - entry_price) * qty

        closed_trades.append(
            {
                "symbol": symbol,
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
    print(f"Average win: ${summary['average_win']:.2f}")
    print(f"Average loss: ${summary['average_loss']:.2f}")
    print(f"Profit factor: {summary['profit_factor']:.2f}")
    print(f"Best symbol: {summary['best_symbol']}")
    print(f"Worst symbol: {summary['worst_symbol']}")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze closed trades from logs/trades.csv")
    parser.add_argument("path", nargs="?", default="logs/trades.csv")
    args = parser.parse_args()

    rows = load_live_trade_log(args.path)
    trades = pair_live_trade_log(rows)
    print_summary(summarize_closed_trades(trades))
