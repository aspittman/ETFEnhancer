"""Offline regression tests: missing market evidence must not liquidate SPY."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import strategy


class RegimeDataSafetyTests(unittest.TestCase):
    def test_missing_invalid_and_failed_data_are_unknown(self):
        for data in (pd.DataFrame(),pd.DataFrame({'Close':[100.]*10}),
                     pd.DataFrame({'Close':[float('nan')]*210}),
                     pd.DataFrame({'Close':[float('inf')]*210})):
            with self.subTest(data=data.shape), patch.object(strategy,'fetch_price_history',return_value=data):
                self.assertIsNone(strategy.get_market_regime('SPY')['is_healthy'])
        with patch.object(strategy,'fetch_price_history',side_effect=RuntimeError('test outage')):
            self.assertIsNone(strategy.get_market_regime('SPY')['is_healthy'])

    def test_confirmed_healthy_and_weak_remain_distinct(self):
        for prices,expected in ((range(1,211),True),(range(210,0,-1),False)):
            with patch.object(strategy,'fetch_price_history',return_value=pd.DataFrame({'Close':list(prices)})):
                self.assertIs(strategy.get_market_regime('SPY')['is_healthy'],expected)

    def test_actual_live_regime_branch_never_sells_unknown(self):
        # Execute the real branch without importing trader or constructing broker clients.
        tree=ast.parse(Path(strategy.__file__).with_name('main.py').read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                    and ast.unparse(n.test)=='switches.market_regime or ENABLE_SPY_CORE')
        code=ast.fix_missing_locations(ast.Module(body=[ast.While(test=ast.Constant(True),body=[branch,ast.Break()],orelse=[])],type_ignores=[]))
        class EndCycle(Exception):pass
        for healthy in (None,False,True):
            for enabled in (False,True):
                with self.subTest(healthy=healthy,filter=enabled):
                    sell=Mock();sleep=Mock(side_effect=EndCycle)
                    namespace=dict(switches=SimpleNamespace(market_regime=enabled),ENABLE_SPY_CORE=True,
                        get_market_regime=Mock(return_value={'is_healthy':healthy,'reason':'test'}),
                        MARKET_REGIME_SYMBOL='SPY',MARKET_REGIME_MA_SHORT=50,MARKET_REGIME_MA_LONG=200,
                        position_state={'SPY':{'position_role':'core','qty':1}},place_trade=sell,
                        print_account_info=Mock(),time=SimpleNamespace(sleep=sleep),SCAN_INTERVAL_SECONDS=300,print=Mock())
                    try:exec(compile(code,'live-regime-branch','exec'),namespace)
                    except EndCycle:pass
                    self.assertEqual(sell.call_count,1 if healthy is False else 0)
                    if healthy is None:self.assertEqual(sleep.call_count,1)
