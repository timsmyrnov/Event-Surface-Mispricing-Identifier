import unittest
from unittest.mock import patch
from esmi.better import Better

class DummySecurities:
    def __init__(self, rows: dict[int, dict]):
        self.rows = rows

    def read_sec(self, sec_id: int):
        return self.rows[sec_id]


class DummyPortfolio:
    def __init__(self):
        self.last = None

    def create_pos(self, sec, edge, vwap, shares):
        self.last = (sec, edge, vwap, shares)
        return self.last


class TestBetter(unittest.TestCase):
    def setUp(self):
        self.sec_good_edge = {
            'id': 1,
            'ticker': 'MSFT',
            'label': '$400-420',
            'url': 'https://example.com/msft',
            'pred_mkt_prob': 0.60,
            'opt_mkt_prob': 0.70,
        }

        self.sec_bad_edge = {
            'id': 2,
            'ticker': 'MSFT',
            'label': '$400-420',
            'url': 'https://example.com/msft',
            'pred_mkt_prob': 0.500,
            'opt_mkt_prob': 0.505,
        }

        self.secs = DummySecurities({
            1: self.sec_good_edge,
            2: self.sec_bad_edge,
        })
        self.portfolio = DummyPortfolio()

        self.better = Better(secs=self.secs, portfolio=self.portfolio)

    def test_compute_edge_above_min(self):
        edge = self.better._compute_edge(1, min_edge=0.05)

        self.assertIsNotNone(edge)
        mag, side = edge
        self.assertAlmostEqual(mag, 0.10)
        self.assertEqual(side, 'Yes')

    def test_compute_edge_below_min(self):
        edge = self.better._compute_edge(2, min_edge=0.05)
        self.assertIsNone(edge)

    @patch('esmi.better.pm.max_invest_for_side_at_price')
    def test_try_bet_success(self, mock_max):
        mock_max.return_value = (100.0, 55.0, 0.55)

        result = self.better.try_bet(1, min_edge=0.05)

        self.assertIsNotNone(self.portfolio.last)
        sec, edge, vwap, shares = self.portfolio.last

        self.assertEqual(sec, self.sec_good_edge)
        self.assertAlmostEqual(vwap, 0.55)
        self.assertAlmostEqual(shares, 100.0)
        self.assertEqual(edge[1], 'Yes')
        self.assertEqual(result, self.portfolio.last)

    @patch('esmi.better.pm.max_invest_for_side_at_price')
    def test_try_bet_fails_when_edge_too_small(self, mock_max):
        mock_max.side_effect = AssertionError(
            'max_invest_for_side_at_price should not be called'
        )

        result = self.better.try_bet(2, min_edge=0.05)

        self.assertEqual(result, 'Bet failed')
        self.assertIsNone(self.portfolio.last)


if __name__ == '__main__':
    unittest.main()
