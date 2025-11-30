from datetime import datetime, timezone
import requests
import json
from urllib.parse import urlparse
from pytickersymbols import PyTickerSymbols
import re

GAMMA_BASE_URL = 'https://gamma-api.polymarket.com'
CLOB_BASE_URL = 'https://clob.polymarket.com'
BASE_URL = 'https://polymarket.com/event/'
SESSION = requests.Session()

stocks = PyTickerSymbols()
TICKERS = {
    s['symbol'].upper()
    for s in stocks.get_all_stocks()
    if s.get('symbol')
}

def get_markets(limit: int = 100, offset: int = 0):
    today_utc = datetime.now(timezone.utc).date()
    end_date_min = today_utc.strftime('%Y-%m-%dT00:00:00Z')
    end_date_max = '2025-12-31T00:00:00Z'

    url = f'{GAMMA_BASE_URL}/markets'
    params = {
        'limit': limit,
        'offset': offset,
        'closed': 'false',
        'order': 'endDate',
        'ascending': 'true',
        'end_date_min': end_date_min,
        'end_date_max': end_date_max
    }

    r = SESSION.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def find_markets_by_keyword(keyword: str, limit: int=100, max_pages: int=40) -> list:
    keyword_lc = keyword.lower()
    matches: list[dict] = []
    offset = 0

    for _ in range(max_pages):
        data = get_markets(limit=limit, offset=offset)

        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = data.get('markets') or data.get('data') or []
        else:
            markets = []

        if not markets:
            break

        for m in markets:
            fields = (
                m.get('question'),
                m.get('title'),
                m.get('eventTitle'),
                m.get('groupItemTitle'),
                m.get('groupItemRange'),
            )

            for f in fields:
                if isinstance(f, str) and keyword_lc in f.lower():
                    matches.append(m)
                    break

        if len(markets) < limit:
            break

        offset += limit

    return matches


def _parse_outcomes(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(o) for o in raw]
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, list):
            return [str(o) for o in decoded]
    except Exception:
        pass
    return [str(raw)]


def _parse_probs(raw):
    vals = _parse_outcomes(raw)
    probs = []
    for v in vals:
        try:
            probs.append(float(v))
        except (TypeError, ValueError):
            continue
    return probs


def get_event_volume(ev: dict) -> float:
    for key in ('volume', 'usdVolume', 'totalVolume', 'volume24h', 'volume24hr'):
        v = ev.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def get_event_expiry(url: str):
    slug = _extract_slug(url)
    if slug is None:
        return None

    resp = requests.get(f'{GAMMA_BASE_URL}/events/slug/{slug}', timeout=10)
    if resp.status_code != 200:
        return None

    event = resp.json()
    markets = event.get('markets') or []
    if not markets:
        return None

    end_str = (
        markets[0].get('endDateIso')
        or markets[0].get('endDate')
        or event.get('endDateIso')
        or event.get('endDate')
    )
    if not end_str:
        return None

    end_str = end_str.replace('Z', '+00:00')
    return datetime.fromisoformat(end_str)


def _extract_slug(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) < 2 or parts[0] not in {'event', 'market'}:
        return None
    return parts[1]


def _get_markets_for_slug(slug: str):
    url_event = f'{GAMMA_BASE_URL}/events/slug/{slug}'
    try:
        r = requests.get(url_event, timeout=10)
        if r.status_code == 200:
            event = r.json()
            markets = event.get('markets') or []
            return markets
        if r.status_code != 404:
            r.raise_for_status()
    except Exception:
        pass

    url_market = f'{GAMMA_BASE_URL}/markets/slug/{slug}'
    try:
        r2 = requests.get(url_market, timeout=10)
        if r2.status_code == 200:
            market = r2.json()
            return [market]
        if r2.status_code != 404:
            r2.raise_for_status()
    except Exception:
        pass

    return []


def _extract_ticker_from_url(url: str) -> str:
    slug = _extract_slug(url)
    return _extract_ticker_from_slug(slug)


def _extract_ticker_from_slug(slug: str, universe: set=TICKERS) -> str:
    tokens = re.split(r'[^A-Za-z0-9\.]+', slug)
    for token in tokens:
        if not token:
            continue
        cand = token.upper()
        if cand in universe:
            return cand

    return None


def load_secs(min_volume: float=12_000.0, keyword: str='close', limit: int=100, max_pages: int=300) -> list:
    markets = find_markets_by_keyword(keyword, limit, max_pages)

    seen = set()
    output = []

    for m in markets:
        events = m.get('events') or []
        for ev in events:
            vol = get_event_volume(ev)
            if vol < min_volume:
                continue

            event_slug = ev.get('slug')
            if not event_slug:
                continue

            ticker = _extract_ticker_from_slug(event_slug)
            if not ticker:
                continue

            url = f'{BASE_URL}{event_slug}'

            bucket_markets = _get_markets_for_slug(event_slug)
            if not bucket_markets:
                continue

            for bm in bucket_markets:
                label = bm.get('groupItemTitle') or bm.get('groupItemRange')

                if 'above' in event_slug:
                    label = '>' + label
                elif 'below' in event_slug:
                    label = '<' + label
                if not label:
                    continue

                key = (event_slug, label)
                if key in seen:
                    continue
                seen.add(key)

                outcomes = _parse_outcomes(bm.get('outcomes'))
                probs = _parse_probs(bm.get('outcomePrices'))

                yes_p = None
                if outcomes and probs:
                    for name, p in zip(outcomes, probs):
                        if str(name).strip().lower() == 'yes':
                            yes_p = p
                            break

                output.append((ticker, label, url, yes_p))

    return output


def _safe_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [v]
    return [v]


def _get_book_for_token(token_id: str) -> dict:
    url = f'{CLOB_BASE_URL}/book'
    r = requests.get(url, params={'token_id': token_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def _pick_market_by_label(markets: list, label: str) -> dict:
    if label is None:
        if not markets:
            raise ValueError('No markets for this slug')
        return markets[0]

    for m in markets:
        m_label = (
            m.get('groupItemTitle')
            or m.get('groupItemRange')
            or m.get('question')
        )
        if m_label == label:
            return m

    raise ValueError(f'Could not find market with label {label!r}')


def max_invest_for_side_at_price(event_url: str, side: str, label: str=None, target_price: float=None):
    slug = _extract_slug(event_url)
    if slug is None:
        raise ValueError('Could not parse slug from URL')

    markets = _get_markets_for_slug(slug)
    if not markets:
        raise ValueError('No markets returned for slug')

    m = _pick_market_by_label(markets, label)

    outcomes = [str(o) for o in _safe_list(m.get('outcomes'))]
    token_ids = [str(t) for t in _safe_list(m.get('clobTokenIds'))]

    if not outcomes or not token_ids or len(outcomes) != len(token_ids):
        raise ValueError('Market is missing outcomes/clobTokenIds')

    norm_side = side.capitalize()
    if norm_side not in outcomes:
        raise ValueError(f'Side {side!r} not found in outcomes {outcomes}')

    idx = outcomes.index(norm_side)
    token_id = token_ids[idx]

    book = _get_book_for_token(token_id)
    asks = book.get('asks') or []
    if not asks:
        return 0.0, 0.0, None

    levels = sorted(
        ((float(a['price']), float(a['size'])) for a in asks),
        key=lambda x: x[0],
    )

    bids = book.get('bids') or []
    if not bids or not asks:
        return 0.0, 0.0, None

    best_ask = min(float(a['price']) for a in asks)
    best_bid = max(float(b['price']) for b in bids)
    current_px = 0.5 * (best_bid + best_ask)

    if target_price is None:
        target_price = current_px + 0.03

    total_size = 0.0
    total_cost = 0.0

    for price, size in levels:
        new_cost = total_cost + price * size
        new_size = total_size + size
        new_avg = new_cost / new_size

        if new_avg <= target_price:
            total_cost = new_cost
            total_size = new_size
        else:
            if price <= target_price:
                break

            numerator = target_price * total_size - total_cost
            denominator = price - target_price
            if denominator > 0 and numerator > 0:
                x = min(size, numerator / denominator)
                total_cost += price * x
                total_size += x
            break
    
    if total_size == 0:
        return 0.0, 0.0, None

    vwap = total_cost / total_size
    return total_size, total_cost, vwap
