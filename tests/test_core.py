import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analytics import (
    add_benchmark_comparison,
    load_live_trade_log,
    pair_live_trade_log,
    summarize_closed_trades,
)
from config import ALPACA_PAPER
from trader import LOG_FILE, TRADING_ENVIRONMENT, wait_for_order_fill
from backtest import (
    BacktestConfig,
    _benchmark_window_return,
    _close_position,
    _manage_spy_core,
    _market_is_healthy,
    _rank_candidates,
)
from strategy import build_strategy_frame, normalize_price_data, signal_from_row
from trader import (
    get_managed_performance,
    get_open_position_symbols,
    get_total_market_value,
    place_trade,
    position_state,
    reconcile_position_state,
    update_midpoint_state,
)
from pivots import new_pivot_state, update_pivot_state, update_structural_stop
from universe import ETF_UNIVERSE, LIVE_TRADING_UNIVERSE, UNIVERSE


class StrategyTests(unittest.TestCase):
    def test_signal_from_row_returns_rankable_candidate(self):
        index = pd.date_range("2026-01-01", periods=260, freq="h")
        close = pd.Series([100 + i * 0.1 for i in range(260)], index=index, dtype=float)
        data = pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
            }
        )

        frame = build_strategy_frame(data, 20, 50, 12, 26, 9, 14)
        signal = signal_from_row("TEST", frame.iloc[-1])

        self.assertIsNotNone(signal)
        self.assertEqual(signal["symbol"], "TEST")
        self.assertGreater(signal["score"], 0)
        self.assertGreater(signal["atr"], 0)

    def test_market_regime_rejects_weak_market(self):
        frame = pd.DataFrame(
            {
                "Close": [90.0],
                "ma_short": [95.0],
                "ma_long": [100.0],
                "prev_ma_short": [96.0],
            },
            index=pd.to_datetime(["2026-06-25"]),
        )

        self.assertFalse(_market_is_healthy(frame.index[-1], frame))

    def test_market_regime_uses_price_above_200_and_50_above_200(self):
        frame = pd.DataFrame(
            {
                "Close": [105.0],
                "ma_short": [101.0],
                "ma_long": [100.0],
                "prev_ma_short": [99.0],
            },
            index=pd.to_datetime(["2026-06-25"]),
        )

        self.assertTrue(_market_is_healthy(frame.index[-1], frame))

    def test_market_regime_accepts_timezone_aware_timestamp(self):
        frame = pd.DataFrame(
            {
                "Close": [105.0],
                "ma_short": [101.0],
                "ma_long": [100.0],
                "prev_ma_short": [99.0],
            },
            index=pd.to_datetime(["2026-06-25"]),
        )
        timestamp = pd.Timestamp("2026-06-25 09:30:00", tz="America/New_York")

        self.assertTrue(_market_is_healthy(timestamp, frame))

    def test_normalize_price_data_removes_index_timezone(self):
        index = pd.date_range(
            "2026-06-25 09:30:00",
            periods=2,
            freq="h",
            tz="America/New_York",
        )
        data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
            },
            index=index,
        )

        clean = normalize_price_data(data)

        self.assertIsNone(clean.index.tz)

    def test_relative_strength_contributes_to_candidate_score(self):
        index = pd.date_range("2026-01-01", periods=260, freq="h")
        close = pd.Series([100 + i * 0.1 for i in range(260)], index=index, dtype=float)
        benchmark_close = pd.Series(list(range(100, 230)) + [229.0] * 130, index=index)
        data = pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": [1000.0] * len(index),
            }
        )
        benchmark = pd.DataFrame(
            {
                "Open": benchmark_close - 0.5,
                "High": benchmark_close + 1.0,
                "Low": benchmark_close - 1.0,
                "Close": benchmark_close,
                "Volume": [1000.0] * len(index),
            }
        )

        frame = build_strategy_frame(data, 20, 50, 12, 26, 9, 14, benchmark)
        signal = signal_from_row("TEST", frame.iloc[-1])

        self.assertIsNotNone(signal)
        self.assertGreater(signal["relative_strength"], 0)
        self.assertIn("volume_score", signal)


class AnalyticsTests(unittest.TestCase):
    def test_load_live_trade_log_accepts_headerless_legacy_log(self):
        contents = (
            "2026-01-01,AAA,buy,2,10,signal,10,,\n"
            "2026-01-02,AAA,sell,2,12,atr_trailing_stop,10,12,4\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.csv"
            path.write_text(contents)

            rows = load_live_trade_log(path)
            trades = pair_live_trade_log(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["timestamp"], "2026-01-01")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["pnl"], 4.0)

    def test_summarize_closed_trades_calculates_expectancy_and_symbols(self):
        summary = summarize_closed_trades(
            [
                {"symbol": "AAA", "pnl": 10.0},
                {"symbol": "AAA", "pnl": -4.0},
                {"symbol": "BBB", "pnl": -2.0},
            ]
        )

        self.assertEqual(summary["total_trades"], 3)
        self.assertAlmostEqual(summary["win_rate"], 1 / 3)
        self.assertAlmostEqual(summary["expectancy"], 4 / 3)
        self.assertEqual(summary["best_symbol"], "AAA")
        self.assertEqual(summary["worst_symbol"], "BBB")

    def test_load_labeled_execution_log_preserves_environment_and_order_id(self):
        contents = (
            "timestamp,environment,order_id,symbol,side,qty,price,reason,entry_price,exit_price,pnl\n"
            "2026-01-01,paper,order-1,AAA,buy,2,10,signal,10,,\n"
            "2026-01-02,paper,order-2,AAA,sell,2,12,stop_loss,10,12,4\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades_paper.csv"
            path.write_text(contents)
            rows = load_live_trade_log(path)
            trades = pair_live_trade_log(rows)

        self.assertEqual(rows[0]["environment"], "paper")
        self.assertEqual(rows[0]["order_id"], "order-1")
        self.assertEqual(trades[0]["environment"], "paper")

    def test_execution_log_matches_configured_environment(self):
        expected = "paper" if ALPACA_PAPER else "live"
        self.assertEqual(TRADING_ENVIRONMENT, expected)
        self.assertEqual(LOG_FILE, f"logs/trades_{expected}.csv")

    def test_benchmark_compares_matching_trade_windows(self):
        trades = [
            {
                "symbol": "AAA",
                "entry_time": "2026-01-02 15:00:00",
                "exit_time": "2026-01-05 15:00:00",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "qty": 1.0,
                "pnl": 10.0,
            },
            {
                "symbol": "BBB",
                "entry_time": "2026-01-05 15:00:00",
                "exit_time": "2026-01-06 15:00:00",
                "entry_price": 50.0,
                "exit_price": 50.0,
                "qty": 2.0,
                "pnl": 0.0,
            },
        ]
        spy = pd.Series(
            [100.0, 105.0, 110.0],
            index=pd.to_datetime(
                ["2026-01-02 15:00:00", "2026-01-05 15:00:00", "2026-01-06 15:00:00"]
            ),
        )

        summary = add_benchmark_comparison(summarize_closed_trades(trades), trades, spy)

        self.assertEqual(summary["benchmark_trade_count"], 2)
        self.assertAlmostEqual(summary["matched_strategy_return"], 0.05)
        self.assertAlmostEqual(summary["matched_benchmark_return"], (0.05 + (110 / 105 - 1)) / 2)
        self.assertAlmostEqual(summary["benchmark_buy_hold_return"], 0.10)
        self.assertEqual(summary["benchmark_outperformance_rate"], 0.5)

    def test_pair_live_trade_log_closes_buy_sell_pairs(self):
        trades = pair_live_trade_log(
            [
                {
                    "timestamp": "2026-01-01",
                    "symbol": "AAA",
                    "side": "buy",
                    "qty": "2",
                    "price": "10",
                    "entry_price": "10",
                },
                {
                    "timestamp": "2026-01-02",
                    "symbol": "AAA",
                    "side": "sell",
                    "qty": "2",
                    "price": "12",
                    "exit_price": "12",
                },
            ]
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["pnl"], 4.0)


class BacktestConfigTests(unittest.TestCase):
    def test_spy_core_enters_at_target_and_exits_on_weak_regime(self):
        timestamp = pd.Timestamp("2026-01-02")
        frames = {
            "SPY": pd.DataFrame(
                {"Close": [100.0]}, index=pd.DatetimeIndex([timestamp])
            )
        }
        positions = {}
        trades = []
        config = BacktestConfig(
            max_total_capital=100.0,
            spy_core_allocation_percent=0.6,
            transaction_cost_bps=0.0,
        )

        _manage_spy_core(timestamp, frames, positions, trades, config, True)
        self.assertEqual(positions["SPY"]["position_role"], "core")
        self.assertAlmostEqual(positions["SPY"]["qty"], 0.6)

        _manage_spy_core(timestamp, frames, positions, trades, config, False)
        self.assertNotIn("SPY", positions)
        self.assertEqual(trades[0]["exit_reason"], "market_regime_exit")

    @patch("backtest.signal_from_row")
    def test_tactical_candidates_must_outperform_spy(self, signal_from_row):
        timestamp = pd.Timestamp("2026-01-02")
        frames = {
            symbol: pd.DataFrame({"Close": [100.0]}, index=[timestamp])
            for symbol in ("SPY", "XLK", "XLP")
        }
        signals = {
            "XLK": {"symbol": "XLK", "price": 100.0, "score": 2.0,
                    "relative_strength": 0.04},
            "XLP": {"symbol": "XLP", "price": 100.0, "score": 4.0,
                    "relative_strength": -0.02},
        }
        signal_from_row.side_effect = lambda symbol, row, switches: signals[symbol]
        pivots = {
            symbol: {"confirmed_swing_low": 90.0}
            for symbol in frames
        }

        candidates = _rank_candidates(
            timestamp, frames, {}, BacktestConfig(), pivots
        )

        self.assertEqual([candidate["symbol"] for candidate in candidates], ["XLK"])

    def test_backtest_config_uses_atr_multiplier(self):
        config = BacktestConfig(atr_multiplier=2.5)
        self.assertEqual(config.atr_multiplier, 2.5)

    def test_midpoint_stop_only_moves_up(self):
        state = {
            "entry_price": 76.0,
            "highest_price_since_entry": 76.0,
            "current_midpoint_stop": 76.0,
        }
        update_midpoint_state(state, 76.0, 82.0)
        self.assertEqual(state["current_midpoint_stop"], 79.0)
        update_midpoint_state(state, 76.0, 78.0)
        self.assertEqual(state["current_midpoint_stop"], 79.0)
        update_midpoint_state(state, 76.0, 100.0)
        self.assertEqual(state["previous_high"], 82.0)
        self.assertEqual(state["current_midpoint_stop"], 91.0)

    def test_transaction_cost_is_deducted_from_backtest_trade(self):
        positions = {
            "AAA": {
                "entry_time": pd.Timestamp("2026-01-01"),
                "entry_price": 100.0,
                "qty": 1.0,
                "entry_score": 1.0,
                "transaction_cost_bps": 5.0,
            }
        }
        trades = []

        _close_position("AAA", positions, trades, pd.Timestamp("2026-01-02"), 110.0, "test")

        self.assertAlmostEqual(trades[0]["gross_pnl"], 10.0)
        self.assertAlmostEqual(trades[0]["estimated_cost"], 0.105)
        self.assertAlmostEqual(trades[0]["pnl"], 9.895)

    def test_benchmark_window_return_uses_matching_dates(self):
        benchmark = pd.DataFrame(
            {"Close": [100.0, 105.0, 110.0]},
            index=pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        )

        result = _benchmark_window_return(
            benchmark, pd.Timestamp("2026-01-15"), pd.Timestamp("2026-03-02")
        )

        self.assertAlmostEqual(result, 110 / 105 - 1)


class TradingAccountTests(unittest.TestCase):
    def test_live_universe_is_exactly_the_etf_universe(self):
        self.assertEqual(UNIVERSE, ETF_UNIVERSE)
        self.assertEqual(LIVE_TRADING_UNIVERSE, frozenset(ETF_UNIVERSE))
        self.assertEqual(len(ETF_UNIVERSE), len(set(ETF_UNIVERSE)))

    @patch("trader.trading_client.submit_order")
    @patch("trader.get_position")
    def test_order_boundary_blocks_non_universe_symbol(self, get_position, submit):
        with patch.dict(position_state, {}, clear=True):
            self.assertFalse(place_trade("NOTETF", "buy", notional=25))
            self.assertFalse(place_trade("BTC/USD", "buy", notional=25))
            self.assertFalse(place_trade("SPY260918C00500000", "buy", notional=25))
        get_position.assert_not_called()
        submit.assert_not_called()

    @patch("trader.get_position", return_value=0)
    def test_local_state_blocks_duplicate_buy_during_broker_lag(self, get_position):
        with patch.dict(
            position_state,
            {"XLP": {"entry_price": 85.0, "qty": 0.25}},
            clear=True,
        ):
            self.assertFalse(place_trade("XLP", "buy", notional=25))

    @patch("trader.save_position_state")
    @patch("trader.alpaca_read")
    def test_reconcile_removes_only_stale_managed_state(self, alpaca_read, save_state):
        alpaca_read.side_effect = [
            [type("Position", (), {"symbol": "XLE"})()],
            [type("Order", (), {"symbol": "XLP"})()],
        ]
        with patch.dict(
            position_state,
            {
                "XLE": {"entry_price": 60.0},
                "XLP": {"entry_price": 85.0},
                "XLV": {"entry_price": 170.0},
                "NOTETF": {"entry_price": 200.0},
            },
            clear=True,
        ):
            self.assertEqual(reconcile_position_state(), ["NOTETF", "XLV"])
            self.assertEqual(set(position_state), {"XLE", "XLP"})
            save_state.assert_called_once()

    @patch("trader.trading_client.get_all_positions")
    def test_open_position_symbols_come_from_actual_positions(self, get_positions):
        get_positions.return_value = [
            type("Position", (), {"symbol": "XLE"})(),
            type("Position", (), {"symbol": "XLRE"})(),
        ]

        with patch.dict(position_state, {"XLE": {"entry_price": 50}}, clear=True):
            self.assertEqual(get_open_position_symbols(), ["XLE"])

    @patch("trader.trading_client.get_all_positions")
    def test_shared_account_positions_do_not_consume_bot_capital(self, get_positions):
        get_positions.return_value = [
            type(
                "Position", (),
                {"symbol": "XLE", "qty": "2", "current_price": "55"},
            )(),
            type(
                "Position", (),
                {"symbol": "NOTETF", "qty": "10", "current_price": "500"},
            )(),
        ]

        with patch.dict(
            position_state,
            {"XLE": {"entry_price": 50, "qty": 0.5}},
            clear=True,
        ):
            self.assertAlmostEqual(get_total_market_value(), 27.5)

    @patch("trader.trading_client.get_all_positions")
    def test_managed_performance_excludes_shared_account_positions(self, get_positions):
        get_positions.return_value = [
            type(
                "Position", (),
                {"symbol": "XLE", "qty": "2", "avg_entry_price": "50", "current_price": "55"},
            )(),
            type(
                "Position", (),
                {"symbol": "NOTETF", "qty": "10", "avg_entry_price": "100", "current_price": "500"},
            )(),
        ]

        with patch.dict(
            position_state,
            {"XLE": {"entry_price": 50, "qty": 0.5}},
            clear=True,
        ):
            cost, value, pnl, percent = get_managed_performance()

        self.assertAlmostEqual(cost, 25.0)
        self.assertAlmostEqual(value, 27.5)
        self.assertAlmostEqual(pnl, 2.5)
        self.assertAlmostEqual(percent, 10.0)

    @patch("trader.get_position", return_value=1)
    def test_bot_refuses_to_sell_unowned_position(self, get_position):
        with patch.dict(position_state, {}, clear=True):
            self.assertFalse(place_trade("NOTETF", "sell", qty=1, reason="test"))

    @patch("trader.time.sleep")
    @patch("trader.alpaca_read")
    def test_wait_for_order_fill_returns_confirmed_fill(self, alpaca_read, sleep):
        accepted = type("Order", (), {"status": "accepted"})()
        filled = type(
            "Order",
            (),
            {"status": "filled", "filled_avg_price": "101.25", "filled_qty": "0.5"},
        )()
        alpaca_read.side_effect = [accepted, filled]

        result = wait_for_order_fill("order-1")

        self.assertIs(result, filled)
        sleep.assert_called_once()


class PivotTests(unittest.TestCase):
    def test_pivots_require_opposite_reversal_without_lookahead(self):
        state = new_pivot_state()
        closes = [100.0, 94.0, 99.0, 101.0, 110.0]
        dates = pd.date_range("2026-01-02", periods=len(closes), freq="W-FRI")
        for date, close in zip(dates, closes):
            event = update_pivot_state(state, close, date, 0.06, 16, 3)

        self.assertIsNone(state["confirmed_swing_high"])
        self.assertEqual(state["confirmed_swing_low"], 94.0)
        self.assertFalse(event["new_pivot_confirmed"])

        event = update_pivot_state(
            state, 103.0, dates[-1] + pd.Timedelta(weeks=1), 0.06, 16, 3
        )
        self.assertTrue(event["new_pivot_confirmed"])
        self.assertEqual(event["pivot_type"], "high")
        self.assertEqual(state["confirmed_swing_high"], 110.0)

    def test_structural_stop_uses_confirmed_pivots_and_never_decreases(self):
        position = {
            "active_structural_low": 76.0,
            "current_structural_stop": None,
        }
        pivots = {"confirmed_swing_low": 76.0, "confirmed_swing_high": 90.0}
        self.assertTrue(update_structural_stop(position, pivots))
        self.assertEqual(position["current_structural_stop"], 83.0)

        pivots = {"confirmed_swing_low": 80.0, "confirmed_swing_high": 90.0}
        self.assertTrue(update_structural_stop(position, pivots))
        self.assertEqual(position["current_structural_stop"], 85.0)

        pivots = {"confirmed_swing_low": 79.0, "confirmed_swing_high": 88.0}
        self.assertFalse(update_structural_stop(position, pivots))
        self.assertEqual(position["current_structural_stop"], 85.0)


if __name__ == "__main__":
    unittest.main()
