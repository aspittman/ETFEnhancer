import unittest

import pandas as pd

from analytics import pair_live_trade_log, summarize_closed_trades
from backtest import BacktestConfig, _market_is_healthy
from strategy import build_strategy_frame, signal_from_row


class StrategyTests(unittest.TestCase):
    def test_signal_from_row_returns_rankable_candidate(self):
        index = pd.date_range("2026-01-01", periods=260, freq="h")
        close = pd.Series(range(100, 360), index=index, dtype=float)
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


class AnalyticsTests(unittest.TestCase):
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
    def test_backtest_config_uses_atr_multiplier(self):
        config = BacktestConfig(atr_multiplier=2.5)
        self.assertEqual(config.atr_multiplier, 2.5)


if __name__ == "__main__":
    unittest.main()
