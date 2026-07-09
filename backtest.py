import argparse
from dataclasses import dataclass

import pandas as pd

from analytics import print_summary, summarize_closed_trades
from config import (
    ATR_TRAILING_MULTIPLIER,
    ATR_WINDOW,
    BLOCKED_SYMBOLS,
    DOLLARS_PER_TRADE,
    ENABLE_ATR_TRAILING_STOP,
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
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MARKET_REGIME_SYMBOL,
    MAX_BUYS_PER_CYCLE,
    MAX_POSITIONS,
    MAX_TOTAL_CAPITAL,
    MA_LONG,
    MA_SHORT,
    MIN_CANDIDATE_SCORE,
    STOP_LOSS_PERCENT,
)
from strategy import (
    StrategySwitches,
    build_strategy_frame,
    fetch_price_history,
    score_strategy_frame,
    signal_from_row,
)
from universe import COMBINED_UNIVERSE, ETF_UNIVERSE, STOCK_UNIVERSE, UNIVERSE


@dataclass
class BacktestConfig:
    period: str = "1y"
    interval: str = "1h"
    dollars_per_trade: float = DOLLARS_PER_TRADE
    max_positions: int = MAX_POSITIONS
    max_total_capital: float = MAX_TOTAL_CAPITAL
    max_buys_per_bar: int = MAX_BUYS_PER_CYCLE
    stop_loss_percent: float = STOP_LOSS_PERCENT
    atr_multiplier: float = ATR_TRAILING_MULTIPLIER
    atr_window: int = ATR_WINDOW
    min_candidate_score: float | None = MIN_CANDIDATE_SCORE
    enable_market_regime_filter: bool = ENABLE_MARKET_REGIME_FILTER
    enable_ma_alignment_filter: bool = ENABLE_MA_ALIGNMENT_FILTER
    enable_macd_filter: bool = ENABLE_MACD_FILTER
    enable_relative_strength_filter: bool = ENABLE_RELATIVE_STRENGTH_FILTER
    enable_momentum_score: bool = ENABLE_MOMENTUM_SCORE
    enable_volume_filter: bool = ENABLE_VOLUME_FILTER
    enable_atr_trend_filter: bool = ENABLE_ATR_TREND_FILTER
    enable_min_score_filter: bool = ENABLE_MIN_SCORE_FILTER
    enable_top_candidate_selection: bool = ENABLE_TOP_CANDIDATE_SELECTION
    enable_atr_trailing_stop: bool = ENABLE_ATR_TRAILING_STOP
    enable_fixed_stop_loss: bool = ENABLE_FIXED_STOP_LOSS

    def switches(self):
        return StrategySwitches(
            market_regime=self.enable_market_regime_filter,
            ma_alignment=self.enable_ma_alignment_filter,
            macd=self.enable_macd_filter,
            relative_strength=self.enable_relative_strength_filter,
            price_momentum=self.enable_momentum_score,
            volume=self.enable_volume_filter,
            atr_trend_quality=self.enable_atr_trend_filter,
            min_score=self.enable_min_score_filter
            and self.min_candidate_score is not None,
            top_candidates=self.enable_top_candidate_selection,
        )


def run_backtest(symbols=None, config=None, blocked_symbols=None, prepared=None):
    config = config or BacktestConfig()
    symbols = list(symbols or UNIVERSE)
    blocked = set(blocked_symbols if blocked_symbols is not None else BLOCKED_SYMBOLS)

    if prepared is None:
        frames, benchmark = _prepare_frames(symbols, config, blocked)
    else:
        frames, benchmark = prepared

    frames = {
        symbol: score_strategy_frame(frame, config.switches())
        for symbol, frame in frames.items()
    }
    if benchmark is not None and not benchmark.empty:
        benchmark = score_strategy_frame(benchmark, config.switches())

    return _simulate_backtest(frames, benchmark, config)


def _prepare_frames(symbols, config, blocked):
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

    relative_strength_benchmark = frames.get(MARKET_REGIME_SYMBOL)
    if relative_strength_benchmark is None:
        data = fetch_price_history(
            MARKET_REGIME_SYMBOL,
            period=config.period,
            interval=config.interval,
        )
        relative_strength_benchmark = build_strategy_frame(
            data,
            MA_SHORT,
            MA_LONG,
            MACD_FAST,
            MACD_SLOW,
            MACD_SIGNAL,
            config.atr_window,
        )

    regime_data = fetch_price_history(MARKET_REGIME_SYMBOL, period=config.period, interval="1d")
    regime_benchmark = build_strategy_frame(
        regime_data,
        MA_SHORT,
        MA_LONG,
        MACD_FAST,
        MACD_SLOW,
        MACD_SIGNAL,
        config.atr_window,
    )

    for symbol, frame in list(frames.items()):
        frames[symbol] = build_strategy_frame(
            frame,
            MA_SHORT,
            MA_LONG,
            MACD_FAST,
            MACD_SLOW,
            MACD_SIGNAL,
            config.atr_window,
            benchmark_frame=relative_strength_benchmark,
        )

    return frames, regime_benchmark


def _simulate_backtest(frames, benchmark, config):
    if not frames:
        return {"trades": [], "summary": summarize_closed_trades([])}

    calendar = sorted(set().union(*(frame.index for frame in frames.values())))
    positions = {}
    closed_trades = []

    for timestamp in calendar:
        _manage_positions(timestamp, frames, positions, closed_trades, config)

        if config.enable_market_regime_filter and not _market_is_healthy(timestamp, benchmark):
            continue

        candidates = _rank_candidates(timestamp, frames, positions, config)
        buys = 0
        for candidate in candidates:
            if len(positions) >= config.max_positions:
                break
            if buys >= config.max_buys_per_bar:
                break
            if _capital_in_use(positions) + config.dollars_per_trade > config.max_total_capital:
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
        if config.enable_atr_trailing_stop and not pd.isna(atr) and float(atr) > 0:
            trail_price = position["highest_price"] - (float(atr) * config.atr_multiplier)

        low = float(row["Low"])
        if config.enable_fixed_stop_loss and low <= stop_price:
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

        signal = signal_from_row(symbol, frame.loc[timestamp], config.switches())
        if not signal:
            continue

        passes_score = (
            not config.enable_min_score_filter
            or config.min_candidate_score is None
            or signal["score"] >= config.min_candidate_score
        )
        if passes_score:
            candidates.append(signal)

    if config.enable_top_candidate_selection:
        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def _capital_in_use(positions):
    return sum(
        position["entry_price"] * position["qty"]
        for position in positions.values()
    )


def _market_is_healthy(timestamp, benchmark):
    if benchmark is None or benchmark.empty:
        return False

    if isinstance(benchmark.index, pd.DatetimeIndex):
        timestamp = pd.Timestamp(timestamp)
        if benchmark.index.tz is None and timestamp.tz is not None:
            timestamp = timestamp.tz_localize(None)
        elif benchmark.index.tz is not None and timestamp.tz is None:
            timestamp = timestamp.tz_localize(benchmark.index.tz)
        elif benchmark.index.tz is not None and timestamp.tz is not None:
            timestamp = timestamp.tz_convert(benchmark.index.tz)

    available = benchmark.index[benchmark.index <= timestamp]
    if len(available) == 0:
        return False

    row = benchmark.loc[available[-1]]
    values = [row.get("Close"), row.get("ma_short"), row.get("ma_long"), row.get("prev_ma_short")]
    if any(pd.isna(value) for value in values):
        return False

    return (
        float(row["Close"]) > float(row["ma_long"])
        and float(row["ma_short"]) > float(row["ma_long"])
    )


def measure_filter_impact(symbols=None, config=None, blocked_symbols=None):
    config = config or BacktestConfig()
    symbols = list(symbols or UNIVERSE)
    blocked = set(blocked_symbols if blocked_symbols is not None else BLOCKED_SYMBOLS)
    prepared = _prepare_frames(symbols, config, blocked)

    baseline = run_backtest(
        symbols=symbols,
        config=config,
        blocked_symbols=blocked,
        prepared=prepared,
    )["summary"]
    rows = [{"filter": "baseline", **_impact_metrics(baseline)}]

    toggles = [
        ("market_regime", "enable_market_regime_filter"),
        ("ma_alignment", "enable_ma_alignment_filter"),
        ("macd", "enable_macd_filter"),
        ("relative_strength", "enable_relative_strength_filter"),
        ("volume", "enable_volume_filter"),
        ("atr_trend_quality", "enable_atr_trend_filter"),
        ("min_score", "enable_min_score_filter"),
        ("top_candidate_selection", "enable_top_candidate_selection"),
        ("atr_trailing_stop", "enable_atr_trailing_stop"),
        ("fixed_stop_loss", "enable_fixed_stop_loss"),
    ]
    for name, field in toggles:
        variant = BacktestConfig(**{**config.__dict__, field: not getattr(config, field)})
        summary = run_backtest(
            symbols=symbols,
            config=variant,
            blocked_symbols=blocked,
            prepared=prepared,
        )["summary"]
        metrics = _impact_metrics(summary)
        metrics["pnl_delta"] = metrics["total_pnl"] - baseline["total_pnl"]
        metrics["expectancy_delta"] = metrics["expectancy"] - baseline["expectancy"]
        metrics["win_rate_delta"] = metrics["win_rate"] - baseline["win_rate"]
        metrics["profit_factor_delta"] = metrics["profit_factor"] - baseline["profit_factor"]
        rows.append({"filter": f"{name}_{'off' if getattr(config, field) else 'on'}", **metrics})

    return rows


def _impact_metrics(summary):
    return {
        "total_pnl": summary["total_pnl"],
        "expectancy": summary["expectancy"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "total_trades": summary["total_trades"],
    }


def print_filter_impact(rows):
    print("===== FILTER IMPACT =====")
    print(
        "filter,total_pnl,expectancy,win_rate,profit_factor,total_trades,"
        "pnl_delta,expectancy_delta,win_rate_delta,profit_factor_delta"
    )
    for row in rows:
        print(
            f"{row['filter']},"
            f"{row['total_pnl']:.2f},"
            f"{row['expectancy']:.2f},"
            f"{row['win_rate']:.2%},"
            f"{row['profit_factor']:.2f},"
            f"{row['total_trades']},"
            f"{row.get('pnl_delta', 0.0):.2f},"
            f"{row.get('expectancy_delta', 0.0):.2f},"
            f"{row.get('win_rate_delta', 0.0):.2%},"
            f"{row.get('profit_factor_delta', 0.0):.2f}"
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


def _symbols_for_universe(name):
    universes = {
        "etf": ETF_UNIVERSE,
        "stock": STOCK_UNIVERSE,
        "combined": COMBINED_UNIVERSE,
    }
    return list(universes[name])


def _run_and_print_report(label, symbols, config, filter_impact=False):
    print(f"\n===== {label.upper()} BACKTEST =====")
    if filter_impact:
        print_filter_impact(measure_filter_impact(symbols=symbols, config=config))
        return

    result = run_backtest(symbols=symbols, config=config)
    print_summary(result["summary"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an offline strategy backtest.")
    parser.add_argument("--symbols", help="Comma-separated symbols. Defaults to universe.py.")
    parser.add_argument(
        "--universe",
        choices=["etf", "stock", "combined", "all"],
        default="etf",
        help="Backtest ETF-only, stock-only, combined, or all reports.",
    )
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--dollars-per-trade", type=float, default=DOLLARS_PER_TRADE)
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument("--max-total-capital", type=float, default=MAX_TOTAL_CAPITAL)
    parser.add_argument("--max-buys-per-bar", type=int, default=MAX_BUYS_PER_CYCLE)
    parser.add_argument("--min-score", type=float, default=MIN_CANDIDATE_SCORE)
    parser.add_argument("--filter-impact", action="store_true")
    parser.add_argument("--disable-market-regime", action="store_true")
    parser.add_argument("--disable-ma-alignment", action="store_true")
    parser.add_argument("--disable-macd", action="store_true")
    parser.add_argument("--disable-relative-strength", action="store_true")
    parser.add_argument("--disable-volume", action="store_true")
    parser.add_argument("--disable-atr-trend", action="store_true")
    parser.add_argument("--disable-min-score", action="store_true")
    parser.add_argument("--disable-top-selection", action="store_true")
    parser.add_argument("--disable-atr-trailing-stop", action="store_true")
    parser.add_argument("--disable-fixed-stop-loss", action="store_true")
    args = parser.parse_args()

    config = BacktestConfig(
        period=args.period,
        interval=args.interval,
        dollars_per_trade=args.dollars_per_trade,
        max_positions=args.max_positions,
        max_total_capital=args.max_total_capital,
        max_buys_per_bar=args.max_buys_per_bar,
        min_candidate_score=args.min_score,
        enable_market_regime_filter=not args.disable_market_regime,
        enable_ma_alignment_filter=not args.disable_ma_alignment,
        enable_macd_filter=not args.disable_macd,
        enable_relative_strength_filter=not args.disable_relative_strength,
        enable_volume_filter=not args.disable_volume,
        enable_atr_trend_filter=not args.disable_atr_trend,
        enable_min_score_filter=not args.disable_min_score,
        enable_top_candidate_selection=not args.disable_top_selection,
        enable_atr_trailing_stop=not args.disable_atr_trailing_stop,
        enable_fixed_stop_loss=not args.disable_fixed_stop_loss,
    )

    if args.symbols:
        symbols = _parse_symbols(args.symbols)
        _run_and_print_report("custom", symbols, config, args.filter_impact)
    elif args.universe == "all":
        for universe_name in ("etf", "stock", "combined"):
            _run_and_print_report(
                universe_name,
                _symbols_for_universe(universe_name),
                config,
                args.filter_impact,
            )
    else:
        _run_and_print_report(
            args.universe,
            _symbols_for_universe(args.universe),
            config,
            args.filter_impact,
        )
