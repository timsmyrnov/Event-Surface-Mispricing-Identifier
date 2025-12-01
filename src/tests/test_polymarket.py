import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
import esmi.polymarket as pm

class TestParseHelpers(unittest.TestCase):
    def test_parse_outcomes_none_returns_empty(self):
        self.assertEqual(pm._parse_outcomes(None), [])

    def test_parse_outcomes_list_to_strs(self):
        raw = [1, 'Yes', True]
        self.assertEqual(pm._parse_outcomes(raw), ['1', 'Yes', 'True'])

    def test_parse_outcomes_json_string_list(self):
        raw = '[Yes, No]'
        self.assertEqual(pm._parse_outcomes(raw), ['[Yes, No]'])

    def test_parse_outcomes_non_list_json_or_raw(self):
        raw = 'foo'
        self.assertEqual(pm._parse_outcomes(raw), ['foo'])

    def test_parse_outcomes_non_json_string(self):
        raw = 'not json'
        self.assertEqual(pm._parse_outcomes(raw), ['not json'])

    def test_parse_probs_mixed_values(self):
        raw = ['0.3', 'abc', 0.7, None]
        self.assertEqual(pm._parse_probs(raw), [0.3, 0.7])


class TestEventVolume(unittest.TestCase):
    def test_get_event_volume_uses_volume_first_if_valid(self):
        ev = {
            'volume': '123.45',
            'usdVolume': '9999',
            'totalVolume': '555',
        }
        self.assertAlmostEqual(pm.get_event_volume(ev), 123.45)

    def test_get_event_volume_skips_invalid_and_uses_next(self):
        ev = {
            'volume': 'not-a-number',
            'usdVolume': None,
            'totalVolume': '10.5',
        }
        self.assertAlmostEqual(pm.get_event_volume(ev), 10.5)

    def test_get_event_volume_returns_zero_when_no_valid_keys(self):
        ev = {
            'volume': 'not-a-number',
            'usdVolume': 'also-bad',
        }
        self.assertEqual(pm.get_event_volume(ev), 0.0)


class TestSlugAndTickerExtraction(unittest.TestCase):
    def test_extract_slug_from_event_url(self):
        url = 'https://polymarket.com/event/msft-close-2025'
        self.assertEqual(pm._extract_slug(url), 'msft-close-2025')

    def test_extract_slug_invalid_path(self):
        url = 'https://polymarket.com/foo/bar'
        self.assertIsNone(pm._extract_slug(url))

    def test_extract_ticker_from_slug_with_custom_universe(self):
        slug = 'what-will-msft-close-at'
        universe = {'MSFT', 'AAPL'}
        self.assertEqual(pm._extract_ticker_from_slug(slug, universe=universe), 'MSFT')

    def test_extract_ticker_returns_none_if_not_found(self):
        slug = 'what-will-stock-close-at'
        universe = {'MSFT', 'AAPL'}
        self.assertIsNone(pm._extract_ticker_from_slug(slug, universe=universe))


class TestSafeListAndBook(unittest.TestCase):
    def test_safe_list_none(self):
        self.assertEqual(pm._safe_list(None), [])

    def test_safe_list_list(self):
        v = [1, 2, 3]
        self.assertIs(pm._safe_list(v), v)

    def test_safe_list_json_string_list(self):
        v = '[1, 2, 3]'
        self.assertEqual(pm._safe_list(v), [1, 2, 3])

    def test_safe_list_non_json_string(self):
        v = 'foo'
        self.assertEqual(pm._safe_list(v), ['foo'])

    def test_safe_list_other_type(self):
        v = 42
        self.assertEqual(pm._safe_list(v), [42])

    @patch('esmi.polymarket.requests.get')
    def test_get_book_for_token_calls_correct_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'asks': [], 'bids': []}
        mock_get.return_value = mock_resp

        token_id = 'abc123'
        book = pm._get_book_for_token(token_id)

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn(pm.CLOB_BASE_URL, args[0])
        self.assertEqual(kwargs['params'], {'token_id': token_id})
        self.assertEqual(book, {'asks': [], 'bids': []})


class TestPickMarketByLabel(unittest.TestCase):
    def test_pick_market_by_label_none_returns_first(self):
        markets = [{'id': 1}, {'id': 2}]
        self.assertEqual(pm._pick_market_by_label(markets, None), markets[0])

    def test_pick_market_by_label_exact_match(self):
        markets = [
            {'groupItemTitle': 'A'},
            {'groupItemRange': 'B'},
            {'question': 'C'},
        ]
        self.assertEqual(pm._pick_market_by_label(markets, 'B'), markets[1])

    def test_pick_market_by_label_raises_when_not_found(self):
        markets = [{'groupItemTitle': 'A'}]
        with self.assertRaises(ValueError):
            pm._pick_market_by_label(markets, 'Z')

    def test_pick_market_by_label_raises_when_empty_and_label_none(self):
        with self.assertRaises(ValueError):
            pm._pick_market_by_label([], None)


class TestGetMarkets(unittest.TestCase):
    @patch('esmi.polymarket.SESSION.get')
    def test_get_markets_calls_session_with_params(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'markets': []}
        mock_get.return_value = mock_resp

        res = pm.get_markets(limit=10, offset=20)
        self.assertEqual(res, {'markets': []})

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn(pm.GAMMA_BASE_URL, args[0])
        self.assertEqual(kwargs['params']['limit'], 10)
        self.assertEqual(kwargs['params']['offset'], 20)
        self.assertEqual(kwargs['timeout'], 10)


class TestFindMarketsByKeyword(unittest.TestCase):
    @patch('esmi.polymarket.get_markets')
    def test_find_markets_by_keyword_matches_fields(self, mock_get_markets):
        mock_get_markets.side_effect = [
            {
                'markets': [
                    {'question': 'Will MSFT close above 400?', 'id': 1},
                    {'title': 'Random event', 'id': 2},
                ]
            },
            {'markets': []},
        ]

        matches = pm.find_markets_by_keyword('msft', limit=2, max_pages=2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['id'], 1)

    @patch('esmi.polymarket.get_markets')
    def test_find_markets_by_keyword_handles_list_response(self, mock_get_markets):
        mock_get_markets.return_value = [
            {'eventTitle': 'MSFT earnings', 'id': 3},
            {'eventTitle': 'Other', 'id': 4},
        ]
        matches = pm.find_markets_by_keyword('earnings', limit=2, max_pages=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['id'], 3)


class TestGetEventExpiry(unittest.TestCase):
    @patch('esmi.polymarket.requests.get')
    def test_get_event_expiry_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'markets': [
                {
                    'endDateIso': '2025-12-31T15:00:00Z',
                }
            ]
        }
        mock_get.return_value = mock_resp

        url = 'https://polymarket.com/event/msft-close-2025'
        dt = pm.get_event_expiry(url)
        expected = datetime(2025, 12, 31, 15, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(dt, expected)

    @patch('esmi.polymarket.requests.get')
    def test_get_event_expiry_returns_none_on_bad_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        url = 'https://polymarket.com/event/msft-close-2025'
        self.assertIsNone(pm.get_event_expiry(url))

    def test_get_event_expiry_returns_none_on_bad_slug(self):
        url = 'https://polymarket.com/foo/bar'
        self.assertIsNone(pm.get_event_expiry(url))


class TestGetMarketsForSlug(unittest.TestCase):
    @patch('esmi.polymarket.requests.get')
    def test_get_markets_for_slug_prefers_event_endpoint(self, mock_get):
        resp_event = MagicMock()
        resp_event.status_code = 200
        resp_event.json.return_value = {'markets': [{'id': 1}]}

        mock_get.return_value = resp_event

        markets = pm._get_markets_for_slug('msft-close-2025')
        self.assertEqual(markets, [{'id': 1}])

    @patch('esmi.polymarket.requests.get')
    def test_get_markets_for_slug_falls_back_to_market_endpoint(self, mock_get):
        resp_event = MagicMock()
        resp_event.status_code = 404

        resp_market = MagicMock()
        resp_market.status_code = 200
        resp_market.json.return_value = {'id': 2}

        def side_effect(url, timeout):
            if 'events/slug' in url:
                return resp_event
            return resp_market

        mock_get.side_effect = side_effect

        markets = pm._get_markets_for_slug('msft-close-2025')
        self.assertEqual(markets, [{'id': 2}])

    @patch('esmi.polymarket.requests.get')
    def test_get_markets_for_slug_returns_empty_on_errors(self, mock_get):
        resp_event = MagicMock()
        resp_event.status_code = 500

        resp_market = MagicMock()
        resp_market.status_code = 500

        def side_effect(url, timeout):
            if 'events/slug' in url:
                return resp_event
            return resp_market

        mock_get.side_effect = side_effect

        markets = pm._get_markets_for_slug('msft-close-2025')
        self.assertEqual(markets, [])


class TestLoadSecs(unittest.TestCase):
    @patch('esmi.polymarket._get_markets_for_slug')
    @patch('esmi.polymarket._extract_ticker_from_slug')
    @patch('esmi.polymarket.find_markets_by_keyword')
    def test_load_secs_basic_flow(
        self, mock_find_markets, mock_extract_ticker, mock_get_markets_for_slug
    ):
        mock_find_markets.return_value = [
            {
                'events': [
                    {
                        'slug': 'msft-above-400',
                        'volume': '20000',
                    }
                ]
            }
        ]

        mock_extract_ticker.return_value = 'MSFT'

        mock_get_markets_for_slug.return_value = [
            {
                'groupItemTitle': '$400-420',
                'outcomes': ['Yes', 'No'],
                'outcomePrices': [0.6, 0.4],
            }
        ]

        secs = pm.load_secs(min_volume=10000.0, keyword='close', limit=10, max_pages=1)
        self.assertEqual(len(secs), 1)

        ticker, label, url, yes_p = secs[0]
        self.assertEqual(ticker, 'MSFT')
        self.assertEqual(label, '>$400-420')
        self.assertTrue(url.startswith(pm.BASE_URL))
        self.assertAlmostEqual(yes_p, 0.6)

    @patch('esmi.polymarket._get_markets_for_slug')
    @patch('esmi.polymarket._extract_ticker_from_slug')
    @patch('esmi.polymarket.find_markets_by_keyword')
    def test_load_secs_filters_by_volume(
        self, mock_find_markets, mock_extract_ticker, mock_get_markets_for_slug
    ):
        mock_find_markets.return_value = [
            {
                'events': [
                    {'slug': 'msft-above-400', 'volume': '5000'},
                ]
            }
        ]

        secs = pm.load_secs(min_volume=10000.0, keyword='close', limit=10, max_pages=1)
        self.assertEqual(secs, [])


class TestMaxInvestForSideAtPrice(unittest.TestCase):
    @patch('esmi.polymarket._get_book_for_token')
    @patch('esmi.polymarket._get_markets_for_slug')
    def test_max_invest_basic_happy_path(
        self, mock_get_markets_for_slug, mock_get_book_for_token
    ):
        mock_get_markets_for_slug.return_value = [
            {
                'groupItemTitle': 'Bucket',
                'outcomes': ['Yes', 'No'],
                'clobTokenIds': ['token_yes', 'token_no'],
            }
        ]

        mock_get_book_for_token.return_value = {
            'asks': [
                {'price': '0.50', 'size': '10'},
                {'price': '0.60', 'size': '20'},
            ],
            'bids': [
                {'price': '0.45', 'size': '5'},
            ],
        }

        url = 'https://polymarket.com/event/msft-close-2025'
        size, cost, vwap = pm.max_invest_for_side_at_price(
            url, side='Yes', label='Bucket', target_price=0.55
        )

        self.assertAlmostEqual(size, 20.0)
        self.assertAlmostEqual(cost, 11.0)
        self.assertAlmostEqual(vwap, 0.55, places=6)

    @patch('esmi.polymarket._get_book_for_token')
    @patch('esmi.polymarket._get_markets_for_slug')
    def test_max_invest_returns_zero_when_no_asks(
        self, mock_get_markets_for_slug, mock_get_book_for_token
    ):
        mock_get_markets_for_slug.return_value = [
            {
                'groupItemTitle': 'Bucket',
                'outcomes': ['Yes', 'No'],
                'clobTokenIds': ['token_yes', 'token_no'],
            }
        ]

        mock_get_book_for_token.return_value = {
            'asks': [],
            'bids': [],
        }

        url = 'https://polymarket.com/event/msft-close-2025'
        size, cost, vwap = pm.max_invest_for_side_at_price(
            url, side='Yes', label='Bucket', target_price=0.55
        )

        self.assertEqual(size, 0.0)
        self.assertEqual(cost, 0.0)
        self.assertIsNone(vwap)

    def test_max_invest_raises_on_bad_url(self):
        bad_url = 'https://polymarket.com/foo/bar'
        with self.assertRaises(ValueError):
            pm.max_invest_for_side_at_price(bad_url, side='Yes')

    @patch('esmi.polymarket._get_markets_for_slug')
    def test_max_invest_raises_when_no_markets(self, mock_get_markets_for_slug):
        mock_get_markets_for_slug.return_value = []
        url = 'https://polymarket.com/event/msft-close-2025'
        with self.assertRaises(ValueError):
            pm.max_invest_for_side_at_price(url, side='Yes')

    @patch('esmi.polymarket._get_markets_for_slug')
    def test_max_invest_raises_when_side_not_found(self, mock_get_markets_for_slug):
        mock_get_markets_for_slug.return_value = [
            {
                'groupItemTitle': 'Bucket',
                'outcomes': ['No'],
                'clobTokenIds': ['token_no'],
            }
        ]
        url = 'https://polymarket.com/event/msft-close-2025'
        with self.assertRaises(ValueError):
            pm.max_invest_for_side_at_price(url, side='Yes', label='Bucket')


if __name__ == '__main__':
    unittest.main()
