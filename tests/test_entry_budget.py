import unittest
from types import SimpleNamespace as N
from unittest.mock import Mock,patch
import trader
import config

class EntryBudgetTests(unittest.TestCase):
    def test_remaining_capital_includes_pending_unfilled_amount(self):
        orders=[N(side='buy',notional='25',filled_qty='1',filled_avg_price='10')]
        with patch.object(trader,'get_total_market_value',return_value=100), \
             patch.object(trader,'alpaca_read',return_value=orders):
            self.assertEqual(trader.get_available_entry_capital(),15)

    def test_unknown_pending_cost_fails_closed(self):
        with patch.object(trader,'get_total_market_value',return_value=100), \
             patch.object(trader,'alpaca_read',return_value=[N(side='buy',notional=None)]):
            with self.assertRaises(RuntimeError):trader.get_available_entry_capital()

    def test_actual_buy_request_fits_remaining_budget(self):
        with patch.object(trader,'position_state',{}), \
             patch.object(trader,'get_position',return_value=0), \
             patch.object(trader,'has_open_order',return_value=False), \
             patch.object(trader,'get_available_entry_capital',return_value=12.349), \
             patch.object(trader.trading_client,'submit_order',side_effect=RuntimeError('test capture')) as submit:
            try:trader.place_trade('SMH','buy',notional=25)
            except RuntimeError:pass
            self.assertEqual(submit.call_args.kwargs.get('order_data',submit.call_args.args[0] if submit.call_args.args else None).notional,12.34)

    def test_dust_balance_does_not_submit(self):
        with patch.object(trader,'position_state',{}), \
             patch.object(trader,'get_position',return_value=0), \
             patch.object(trader,'has_open_order',return_value=False), \
             patch.object(trader,'get_available_entry_capital',return_value=2), \
             patch.object(trader.trading_client,'submit_order') as submit:
            self.assertFalse(trader.place_trade('SMH','buy',notional=25));submit.assert_not_called()
