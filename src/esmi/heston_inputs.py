import datetime as dt
from dataclasses import dataclass
import numpy as np
import yfinance as yf
import esmi.market_data as md

@dataclass
class HestonInputs:
    ticker: str
    expiry: str
    S0: float
    tau: float
    r: float
    strikes: np.ndarray
    call_mid: np.ndarray
    put_mid: np.ndarray
    atm_iv: float

def get_heston_inputs(ticker: str, expiry: str, risk_free_rate: float=None) -> HestonInputs:
    tk = yf.Ticker(ticker)
    hist = tk.history(period='1d')
    if hist.empty:
        raise ValueError(f'No price history for {ticker}')

    S0 = float(hist['Close'].iloc[-1])

    today = dt.datetime.now(dt.timezone.utc).date()
    req_exp_dt = dt.datetime.strptime(expiry, '%Y-%m-%d').date()

    available_expiries = list(getattr(tk, 'options', []))
    if not available_expiries:
        raise ValueError(f'No listed options expiries for {ticker}')

    available_dates = [
        dt.datetime.strptime(e, '%Y-%m-%d').date() for e in available_expiries
    ]

    if expiry in available_expiries:
        chosen_expiry_date = req_exp_dt
        chosen_expiry_str = expiry
    else:
        future_candidates = [
            d for d in available_dates
            if d >= req_exp_dt and d > today
        ]

        if not future_candidates:
            future_candidates = [d for d in available_dates if d > today]

        if not future_candidates:
            raise ValueError(f'No future expiries available for {ticker}')

        chosen_expiry_date = min(future_candidates)
        chosen_expiry_str = chosen_expiry_date.strftime('%Y-%m-%d')

    days_to_expiry = (chosen_expiry_date - today).days
    if days_to_expiry <= 0:
        raise ValueError('Expiry is not in the future')

    tau = days_to_expiry / 365.0

    if risk_free_rate is not None:
        r = float(risk_free_rate)
    else:
        r = md.get_latest_risk_free_rate()

    opt_chain = tk.option_chain(chosen_expiry_str)
    calls = opt_chain.calls.copy()
    puts = opt_chain.puts.copy()

    def mid_price(df):
        bid = df.get('bid', np.nan)
        ask = df.get('ask', np.nan)
        last = df.get('lastPrice', np.nan)
        mid = (bid + ask) / 2.0
        mid = np.where(np.isnan(mid), last, mid)
        return mid.astype(float)

    call_mid = mid_price(calls)
    put_mid = mid_price(puts)
    strikes = calls['strike'].values.astype(float)

    atm_iv = None
    if 'impliedVolatility' in calls.columns:
        idx_atm = np.argmin(np.abs(strikes - S0))
        atm_iv = float(calls['impliedVolatility'].iloc[idx_atm])

    return HestonInputs(
        ticker=ticker,
        expiry=chosen_expiry_str,
        S0=S0,
        tau=tau,
        r=r,
        strikes=strikes,
        call_mid=call_mid,
        put_mid=put_mid,
        atm_iv=atm_iv,
    )
