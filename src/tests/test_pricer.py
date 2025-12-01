import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from esmi.pricer import Pricer

class DummyModel:
    def __init__(self):
        self.S0 = 100.0
        self.tau = 1.0
        self.r = 0.0
        self.K_min = 0.0
        self.K_max = 2.0
        self.dK = 1.0
        self.strikes = None
        self.call_prices = None
        self.bl_pdf = None

    def heston_price_rec(self, K):
        return 0.5 * K * K


class TestParsePriceRange(unittest.TestCase):
    def setUp(self):
        self.pricer = Pricer(secs=MagicMock())

    def test_parse_price_range_simple_dash(self):
        lo, hi = self.pricer._parse_price_range('$400-420')
        self.assertEqual(lo, 400.0)
        self.assertEqual(hi, 420.0)

    def test_parse_price_range_unicode_dash(self):
        label = '$400\u2013420'
        lo, hi = self.pricer._parse_price_range(label)
        self.assertEqual(lo, 400.0)
        self.assertEqual(hi, 420.0)

    def test_parse_price_range_less_than(self):
        lo, hi = self.pricer._parse_price_range('<$400')
        self.assertEqual(lo, 0)
        self.assertEqual(hi, 400.0)

    def test_parse_price_range_greater_than(self):
        lo, hi = self.pricer._parse_price_range('>$400')
        self.assertEqual(lo, 400.0)
        self.assertEqual(hi, 100_000_000)

    def test_parse_price_range_raises_on_invalid(self):
        with self.assertRaises(ValueError):
            self.pricer._parse_price_range('MSFT close')


class TestComputeCallCurve(unittest.TestCase):
    def setUp(self):
        self.pricer = Pricer(secs=MagicMock())

    def test_compute_call_curve_sets_strikes_and_call_prices(self):
        model = DummyModel()
        model.K_min = 100.0
        model.K_max = 102.0
        model.dK = 1.0

        df = self.pricer._compute_call_curve(model)

        expected_strikes = np.array([100.0, 101.0, 102.0])
        expected_calls = np.array([0.5 * k * k for k in expected_strikes])

        self.assertTrue(np.allclose(model.strikes, expected_strikes))
        self.assertTrue(np.allclose(model.call_prices, expected_calls))
        self.assertTrue(np.allclose(df['strike'].values, expected_strikes))
        self.assertTrue(np.allclose(df['call'].values, expected_calls))


class TestBreedenLitzenberger(unittest.TestCase):
    def setUp(self):
        self.pricer = Pricer(secs=MagicMock())

    def test_breeden_litzenberger_uses_existing_call_curve(self):
        model = DummyModel()
        model.r = 0.0
        model.tau = 1.0
        model.dK = 1.0
        model.strikes = np.array([0.0, 1.0, 2.0])
        model.call_prices = np.array([0.0, 0.5, 2.0])

        df = self.pricer._breeden_litzenberger_pdf(model)

        self.assertIn('strike', df.columns)
        self.assertIn('call', df.columns)
        self.assertIn('bl_pdf', df.columns)

        bl = model.bl_pdf
        self.assertTrue(np.isnan(bl[0]))
        self.assertTrue(np.isnan(bl[2]))
        self.assertAlmostEqual(bl[1], 1.0, places=6)

    @patch.object(Pricer, '_compute_call_curve')
    def test_breeden_litzenberger_calls_compute_call_curve_when_missing(self, mock_curve):
        model = DummyModel()
        model.call_prices = None
        model.strikes = None
        model.dK = 1.0
        model.r = 0.0
        model.tau = 1.0

        def side_effect(m):
            m.strikes = np.array([0.0, 1.0, 2.0])
            m.call_prices = np.array([0.0, 0.5, 2.0])
            return pd.DataFrame({'strike': [0.0, 1.0, 2.0], 'call': [0.0, 0.5, 2.0]})

        mock_curve.side_effect = side_effect

        self.pricer._breeden_litzenberger_pdf(model)

        mock_curve.assert_called_once_with(model)


class TestProbBetween(unittest.TestCase):
    def setUp(self):
        self.pricer = Pricer(secs=MagicMock())

    def test_prob_between_no_overlap_returns_zero(self):
        model = DummyModel()
        model.strikes = np.array([0.0, 1.0, 2.0])
        model.bl_pdf = np.array([0.0, 1.0, 0.0])

        p = self.pricer.prob_between(model, 3.0, 4.0, renormalize=False)
        self.assertEqual(p, 0.0)

    def test_prob_between_without_renormalization(self):
        model = DummyModel()
        model.strikes = np.array([0.0, 1.0, 2.0])
        model.bl_pdf = np.array([0.0, 1.0, 0.0])

        p = self.pricer.prob_between(model, 0.0, 1.0, renormalize=False)
        self.assertAlmostEqual(p, 0.5, places=6)

    def test_prob_between_with_renormalization(self):
        model = DummyModel()
        model.strikes = np.array([0.0, 1.0, 2.0])
        model.bl_pdf = np.array([0.0, 1.0, 0.0])

        p = self.pricer.prob_between(model, 0.0, 1.0, renormalize=True)
        self.assertAlmostEqual(p, 0.5, places=6)

    @patch.object(Pricer, '_breeden_litzenberger_pdf')
    def test_prob_between_calls_bl_when_missing_pdf(self, mock_bl):
        model = DummyModel()
        model.strikes = None
        model.bl_pdf = None

        mock_bl.side_effect = lambda m: setattr(m, 'bl_pdf', np.array([0.0, 1.0, 0.0])) or setattr(m, 'strikes', np.array([0.0, 1.0, 2.0]))

        p = self.pricer.prob_between(model, 0.0, 2.0, renormalize=False)
        mock_bl.assert_called_once_with(model)
        self.assertGreater(p, 0.0)


class TestComputeHestonProb(unittest.TestCase):
    @patch('esmi.pricer.pm.get_event_expiry')
    @patch('esmi.pricer.hi.get_heston_inputs')
    @patch('esmi.pricer.Heston')
    def test_compute_heston_prob_calls_dependencies(self, mock_Heston, mock_get_inputs, mock_get_expiry):
        secs = MagicMock()
        secs.read_sec.return_value = {
            'ticker': 'MSFT',
            'label': '$400-420',
            'url': 'https://polymarket.com/event/msft-close-2025',
        }

        class Inputs:
            def __init__(self):
                self.ticker = 'MSFT'
                self.expiry = '2025-12-31'
                self.S0 = 100.0
                self.tau = 0.5
                self.r = 0.02
                self.strikes = np.array([90.0, 100.0, 110.0])
                self.call_mid = np.array([10.0, 5.0, 2.0])
                self.put_mid = np.array([2.0, 5.0, 10.0])
                self.atm_iv = 0.2

        mock_get_inputs.return_value = Inputs()

        from datetime import datetime, timezone
        mock_get_expiry.return_value = datetime(2025, 12, 31, tzinfo=timezone.utc)

        heston_instance = MagicMock()
        mock_Heston.return_value = heston_instance

        pricer = Pricer(secs=secs)

        with patch.object(Pricer, '_breeden_litzenberger_pdf') as mock_bl, \
             patch.object(Pricer, 'prob_between') as mock_prob_between:
            mock_prob_between.return_value = 0.42

            prob = pricer.compute_heston_prob(sec_id=1)

            secs.read_sec.assert_called_once_with(1)
            mock_get_expiry.assert_called_once_with('https://polymarket.com/event/msft-close-2025')
            mock_get_inputs.assert_called_once_with('MSFT', '2025-12-31')
            mock_Heston.assert_called_once()
            mock_bl.assert_called_once_with(heston_instance)
            mock_prob_between.assert_called_once()
            args, kwargs = mock_prob_between.call_args
            self.assertIs(args[0], heston_instance)
            self.assertAlmostEqual(args[1], 400.0)
            self.assertAlmostEqual(args[2], 420.0)
            self.assertTrue(kwargs.get('renormalize'))
            self.assertAlmostEqual(prob, 0.42, places=6)


if __name__ == '__main__':
    unittest.main()
