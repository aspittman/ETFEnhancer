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
    ENABLE_DYNAMIC_MIDPOINT_STOP,
    ENABLE_STRUCTURAL_MIDPOINT_STOP,
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
    PIVOT_LOOKBACK_WEEKS,
    PIVOT_REVERSAL_PERCENT,
    MIN_WEEKS_BETWEEN_PIVOTS,
    USE_TENTATIVE_HIGH_FOR_STOP,
    STRUCTURAL_STOP_EXIT_TIMEFRAME,
    PIVOT_TIMEFRAME,
    PIVOT_PRICE_SOURCE,
)
from pivots import completed_weekly_closes, new_pivot_state, update_pivot_state, update_structural_stop
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
    enable_dynamic_midpoint_stop: bool = ENABLE_DYNAMIC_MIDPOINT_STOP
    enable_structural_midpoint_stop: bool = ENABLE_STRUCTURAL_MIDPOINT_STOP
    enable_fixed_stop_loss: bool = ENABLE_FIXED_STOP_LOSS
    enable_bullish_candle_confirmation: bool = ENABLE_BULLISH_CANDLE_CONFIRMATION
    pivot_reversal_percent: float = PIVOT_REVERSAL_PERCENT
    pivot_lookback_weeks: int = PIVOT_LOOKBACK_WEEKS
    min_weeks_between_pivots: int = MIN_WEEKS_BETWEEN_PIVOTS
    use_tentative_high_for_stop: bool = USE_TENTATIVE_HIGH_FOR_STOP
    structural_stop_exit_timeframe: str = STRUCTURAL_STOP_EXIT_TIMEFRAME
    pivot_timeframe: str = PIVOT_TIMEFRAME
    pivot_price_source: str = PIVOT_PRICE_SOURCE

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
            bullish_candle=self.enable_bullish_candle_confirmation,
        )


def run_backtest(symbols=None, config=None, blocked_symbols=None, prepared=None):
    config = config or BacktestConfig()
    symbols = list(symbols or UNIVERSE)
    blocked = set(blocked_symbols if blocked_symbols is not None else BLOCKED_SYMBOLS)

    if prepared is None:
        frames, benchmark, pivot_daily = _prepare_frames(symbols, config, blocked)
    else:
        frames, benchmark, pivot_daily = prepared

    frames = {
        symbol: score_strategy_frame(frame, config.switches())
        for symbol, frame in frames.items()
    }
    if benchmark is not None and not benchmark.empty:
        benchmark = score_strategy_frame(benchmark, config.switches())

    return _simulate_backtest(frames, benchmark, pivot_daily, config)


def _prepare_frames(symbols, config, blocked):
    frames = {}
    pivot_daily = {}
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
            daily = fetch_price_history(symbol, period=config.period, interval="1d")
            if not daily.empty:
                pivot_daily[symbol] = daily

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

    if MARKET_REGIME_SYMBOL in frames and MARKET_REGIME_SYMBOL not in pivot_daily:
        pivot_daily[MARKET_REGIME_SYMBOL] = regime_data
    return frames, regime_benchmark, pivot_daily


def _simulate_backtest(frames, benchmark, pivot_daily, config):
    if not frames:
        return {"trades": [], "summary": summarize_closed_trades([])}

    calendar = sorted(set().union(*(frame.index for frame in frames.values())))
    positions = {}
    closed_trades = []
    pivot_states = {symbol: new_pivot_state() for symbol in frames}
    weekly_closes = {
        symbol: completed_weekly_closes(
            pivot_daily.get(symbol, frame),
            timeframe=config.pivot_timeframe,
            price_source=config.pivot_price_source,
        )
        for symbol, frame in frames.items()
    }
    pivot_cursors = {symbol: 0 for symbol in frames}

    for timestamp in calendar:
        _advance_backtest_pivots(
            timestamp, weekly_closes, pivot_cursors, pivot_states, config
        )
        _manage_positions(
            timestamp, frames, positions, closed_trades, config, pivot_states
        )

        if config.enable_market_regime_filter and not _market_is_healthy(timestamp, benchmark):
            continue

        candidates = _rank_candidates(timestamp, frames, positions, config, pivot_states)
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
            anchor = pivot_states[symbol].get("confirmed_swing_low")
            if anchor is None:
                continue
            qty = config.dollars_per_trade / price
            positions[symbol] = {
                "symbol": symbol,
                "entry_time": timestamp,
                "entry_price": price,
                "qty": qty,
                "highest_price": price,
                "previous_high": price,
                "current_midpoint_stop": price,
                "trade_anchor_low": float(anchor),
                "active_structural_low": float(anchor),
                "current_structural_stop": None,
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

    summary = summarize_closed_trades(closed_trades)
    summary["total_return"] = summary["total_pnl"] / config.max_total_capital if config.max_total_capital else 0.0
    pivot_count = sum(int(state.get("confirmed_pivot_count", 0)) for state in pivot_states.values())
    confirmation_weeks = sum(float(state.get("total_confirmation_weeks", 0)) for state in pivot_states.values())
    summary["confirmed_pivots"] = pivot_count
    summary["average_pivot_confirmation_weeks"] = confirmation_weeks / pivot_count if pivot_count else 0.0
    return {
        "trades": closed_trades,
        "summary": summary,
    }


def _advance_backtest_pivots(timestamp, weekly_closes, cursors, states, config):
    now = pd.Timestamp(timestamp)
    if now.tzinfo is not None:
        now = now.tz_localize(None)
    for symbol, weekly in weekly_closes.items():
        items = list(weekly.items())
        cursor = cursors[symbol]
        # A Friday-labelled close is available after Friday has completed.
        while cursor < len(items) and pd.Timestamp(items[cursor][0]) + pd.Timedelta(days=1) <= now:
            date, close = items[cursor]
            update_pivot_state(
                states[symbol], close, date, config.pivot_reversal_percent,
                config.pivot_lookback_weeks, config.min_weeks_between_pivots,
            )
            cursor += 1
        cursors[symbol] = cursor


def _manage_positions(timestamp, frames, positions, closed_trades, config, pivot_states):
    for symbol in list(positions):
        frame = frames.get(symbol)
        if frame is None or timestamp not in frame.index:
            continue

        row = frame.loc[timestamp]
        if pd.isna(row.get("Close")) or pd.isna(row.get("Low")) or pd.isna(row.get("High")):
            continue

        position = positions[symbol]
        update_structural_stop(
            position, pivot_states[symbol], config.use_tentative_high_for_stop
        )
        pivot_states[symbol]["current_structural_stop"] = position.get(
            "current_structural_stop"
        )
        observed_high = float(row["High"])
        if observed_high > position["highest_price"]:
            old_high = position["highest_price"]
            position["previous_high"] = old_high
            position["highest_price"] = observed_high
            position["current_midpoint_stop"] = max(
                position["current_midpoint_stop"],
                (old_high + observed_high) / 2,
            )

        stop_price = position["entry_price"] * (1 - config.stop_loss_percent)
        atr = row.get("atr")
        trail_price = None
        if config.enable_atr_trailing_stop and not pd.isna(atr) and float(atr) > 0:
            trail_price = position["highest_price"] - (float(atr) * config.atr_multiplier)

        low = float(row["Low"])
        close = float(row["Close"])
        structural_close = _is_exit_close(timestamp, frame, config.structural_stop_exit_timeframe)
        if config.enable_fixed_stop_loss and low <= stop_price:
            _close_position(
                symbol,
                positions,
                closed_trades,
                timestamp,
                stop_price,
                "stop_loss",
            )
        elif (
            config.enable_structural_midpoint_stop
            and structural_close
            and position.get("current_structural_stop") is not None
            and close < position["current_structural_stop"]
        ):
            _close_position(
                symbol,
                positions,
                closed_trades,
                timestamp,
                close,
                "structural_midpoint_stop",
            )
        elif config.enable_dynamic_midpoint_stop and close < position["current_midpoint_stop"]:
            _close_position(
                symbol,
                positions,
                closed_trades,
                timestamp,
                close,
                "dynamic_midpoint_stop",
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


def _rank_candidates(timestamp, frames, positions, config, pivot_states):
    candidates = []
    for symbol, frame in frames.items():
        if symbol in positions or timestamp not in frame.index:
            continue
        anchor = pivot_states[symbol].get("confirmed_swing_low")
        if anchor is None:
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
            signal["structural_low_distance"] = (signal["price"] - float(anchor)) / float(anchor)
            candidates.append(signal)

    if config.enable_top_candidate_selection:
        candidates.sort(
            key=lambda candidate: (
                candidate["structural_low_distance"],
                -candidate["score"],
            )
        )
    return candidates


def _is_exit_close(timestamp, frame, timeframe):
    timestamp = pd.Timestamp(timestamp)
    same_day = frame.index[pd.DatetimeIndex(frame.index).date == timestamp.date()]
    if len(same_day) == 0 or timestamp != same_day[-1]:
        return False
    if timeframe == "1d":
        return True
    if timeframe == "1wk":
        return timestamp.weekday() == 4
    return True


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
        ("dynamic_midpoint_stop", "enable_dynamic_midpoint_stop"),
        ("structural_midpoint_stop", "enable_structural_midpoint_stop"),
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


def compare_exit_strategies(label, symbols, config):
    print(f"\n===== {label.upper()} EXIT COMPARISON =====")
    blocked = set(BLOCKED_SYMBOLS)
    prepared = _prepare_frames(symbols, config, blocked)
    for name, dynamic_enabled, structural_enabled, atr_enabled in (
        ("per_price_midpoint", True, False, False),
        ("weekly_structural_midpoint", False, True, False),
        ("atr_trailing", False, False, True),
    ):
        variant = BacktestConfig(
            **{
                **config.__dict__,
                "enable_dynamic_midpoint_stop": dynamic_enabled,
                "enable_structural_midpoint_stop": structural_enabled,
                "enable_atr_trailing_stop": atr_enabled,
            }
        )
        print(f"\n--- {name} ---")
        print_summary(
            run_backtest(
                symbols=symbols,
                config=variant,
                blocked_symbols=blocked,
                prepared=prepared,
            )["summary"]
        )


def run_pivot_grid(label, symbols, config):
    print(f"\n===== {label.upper()} STRUCTURAL PIVOT GRID =====")
    blocked = set(BLOCKED_SYMBOLS)
    prepared = _prepare_frames(symbols, config, blocked)
    for reversal in (0.04, 0.06, 0.08, 0.10):
        for lookback in (12, 16, 26):
            variant = BacktestConfig(
                **{
                    **config.__dict__,
                    "pivot_reversal_percent": reversal,
                    "pivot_lookback_weeks": lookback,
                    "enable_dynamic_midpoint_stop": False,
                    "enable_structural_midpoint_stop": True,
                    "enable_atr_trailing_stop": False,
                }
            )
            print(f"\n--- reversal={reversal:.0%} lookback={lookback}w ---")
            print_summary(
                run_backtest(
                    symbols=symbols, config=variant, blocked_symbols=blocked,
                    prepared=prepared,
                )["summary"]
            )


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
    parser.add_argument("--compare-exits", action="store_true")
    parser.add_argument("--pivot-grid", action="store_true")
    parser.add_argument("--pivot-reversal", type=float, default=PIVOT_REVERSAL_PERCENT)
    parser.add_argument("--pivot-lookback", type=int, default=PIVOT_LOOKBACK_WEEKS)
    parser.add_argument("--disable-market-regime", action="store_true")
    parser.add_argument("--disable-ma-alignment", action="store_true")
    parser.add_argument("--disable-macd", action="store_true")
    parser.add_argument("--disable-relative-strength", action="store_true")
    parser.add_argument("--disable-volume", action="store_true")
    parser.add_argument("--disable-atr-trend", action="store_true")
    parser.add_argument("--disable-min-score", action="store_true")
    parser.add_argument("--disable-top-selection", action="store_true")
    parser.add_argument("--disable-atr-trailing-stop", action="store_true")
    parser.add_argument("--disable-midpoint-stop", action="store_true")
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
        enable_atr_trailing_stop=ENABLE_ATR_TRAILING_STOP and not args.disable_atr_trailing_stop,
        enable_dynamic_midpoint_stop=ENABLE_DYNAMIC_MIDPOINT_STOP and not args.disable_midpoint_stop,
        enable_structural_midpoint_stop=ENABLE_STRUCTURAL_MIDPOINT_STOP and not args.disable_midpoint_stop,
        enable_fixed_stop_loss=not args.disable_fixed_stop_loss,
        pivot_reversal_percent=args.pivot_reversal,
        pivot_lookback_weeks=args.pivot_lookback,
    )

    if args.symbols:
        symbols = _parse_symbols(args.symbols)
        if args.pivot_grid:
            run_pivot_grid("custom", symbols, config)
        elif args.compare_exits:
            compare_exit_strategies("custom", symbols, config)
        else:
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
        symbols = _symbols_for_universe(args.universe)
        if args.pivot_grid:
            run_pivot_grid(args.universe, symbols, config)
        elif args.compare_exits:
            compare_exit_strategies(args.universe, symbols, config)
        else:
            _run_and_print_report(args.universe, symbols, config, args.filter_impact)
