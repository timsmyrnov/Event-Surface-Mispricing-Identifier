import numpy as np
import yfinance as yf
from datetime import datetime as dt, timezone

def get_latest_close_price(ticker: str) -> float:
    ticker = yf.Ticker(ticker) if type(ticker) is str else ticker
    hist = ticker.history(period='1d')
    if hist.empty:
        raise RuntimeError('No price history returned from yfinance.')

    price = float(hist['Close'].iloc[-1])
    return price


def get_call_prices(ticker: str, expiry: str, strikes: np.ndarray) -> np.ndarray:
    ticker = yf.Ticker(ticker)
    opt_chain = ticker.option_chain(expiry)
    all_calls = opt_chain.calls.copy()

    all_expiries = list(getattr(ticker, 'options', []))
    if expiry not in all_expiries:
        raise ValueError(f'Expiry {expiry} not found in the options chain')
    
    call_prices = []
    for strike in strikes:
        call_prices.append(all_calls.loc[all_calls['strike'] == strike])

    return np.array(call_prices)


def get_strikes(ticker: str, expiry: str, side: str='BUY') -> np.ndarray:
    ticker = yf.Ticker(ticker)
    opt_chain = ticker.option_chain(expiry)

    if side.upper() == 'BUY':
        calls = opt_chain.calls.copy()
        strikes = calls['strike'].values.astype(float)
    elif side.upper() == 'SELL':
        puts = opt_chain.puts.copy()
        strikes = puts['strike'].values.astype(float)
    else:
        raise ValueError(f'Side can only be BUY/SELL')
    
    return strikes


def get_latest_risk_free_rate() -> float:
    irx = yf.Ticker('^IRX')
    hist = irx.history(period='5d')
    if hist.empty:
        raise ValueError(f'No IRX history found')

    last = hist['Close'].iloc[-1]
    return float(last) / 100.0


def get_closest_expiry(ticker: str, expiry: str) -> str:
    ticker = yf.Ticker(ticker)
    today = dt.now(timezone.utc).date()

    try:
        req_expiry = dt.strptime(expiry, '%Y-%m-%d').date()
    except Exception as e:
        raise ValueError(f'Invalid expiry {expiry!r}, expected YYYY-MM-DD') from e

    all_expiries = list(getattr(ticker, 'options', []))
    if not all_expiries:
        raise ValueError(f'No listed options expiries for {ticker}')

    all_dates = [
        dt.strptime(e, '%Y-%m-%d').date() for e in all_expiries
    ]

    if expiry in all_expiries:
        chosen_date = req_expiry

    else:
        candidates = [
            d for d in all_dates
            if d >= req_expiry and d > today
        ]

        if not candidates:
            candidates = [d for d in all_dates if d > today]
        if not candidates:
            raise ValueError(f'No future expiries available for {ticker}')

        chosen_date = min(candidates)

    expiry_date = chosen_date
    expiry_str = chosen_date.strftime('%Y-%m-%d')

    days_to_expiry = (expiry_date - today).days
    if days_to_expiry <= 0:
        raise ValueError('Expiry is not in the future')

    return expiry_str


def get_atm_data(ticker: str, expiry: str) -> tuple[float, float]:
    ticker = yf.Ticker(ticker)
    price = get_latest_close_price(ticker)

    opt = ticker.option_chain(expiry)
    calls = opt.calls

    if calls.empty:
        raise RuntimeError(f'No call options for {ticker} on {expiry}')

    valid = calls.dropna(subset=['impliedVolatility'])
    if valid.empty:
        raise RuntimeError(f'No valid IV data for {ticker} on {expiry}')

    atm_idx = (valid['strike'] - price).abs().idxmin()
    row = valid.loc[atm_idx]

    atm_strike = float(row['strike'])
    ann_iv = float(row['impliedVolatility'])

    return atm_strike, ann_iv
