from datetime import datetime as dt, timezone
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
    atm_iv: float

def get_heston_inputs(ticker: str, expiry: str, risk_free_rate: float=None) -> HestonInputs:
    S0 = md.get_latest_close_price(ticker)

    today = dt.now(timezone.utc).date()
    expiry = md.get_closest_expiry(ticker, expiry)
    expiry_date = dt.strptime(expiry, '%Y-%m-%d').date()
    ticker = yf.Ticker(ticker)

    days_to_expiry = (expiry_date - today).days
    if days_to_expiry <= 0:
        raise ValueError('Expiry is not in the future')

    tau = days_to_expiry / 365.0

    if risk_free_rate is not None:
        r = float(risk_free_rate)
    else:
        r = md.get_latest_risk_free_rate()

    strikes = md.get_strikes(ticker, expiry, 'BUY')
    _, atm_iv = md.get_atm_data(ticker, expiry)

    return HestonInputs(
        ticker=ticker,
        expiry=expiry,
        S0=S0,
        tau=tau,
        r=r,
        strikes=strikes,
        atm_iv=atm_iv
    )


if __name__ == '__main__':
    mkt_data = get_heston_inputs('NVDA', '2026-01-01')
    print(mkt_data, mkt_data.S0, mkt_data.tau, mkt_data.r)
