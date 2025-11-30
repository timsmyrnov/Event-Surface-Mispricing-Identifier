import re
import numpy as np
import pandas as pd
from secs import Securities
from black_scholes import BlackScholes
from heston import Heston
import heston_inputs as hi
import polymarket as pm

class Pricer:
    def __init__(self):
        self.secs = Securities()

    def compute_black_scholes_prob(self, sec_id: int) -> float:
        bs_model = BlackScholes(
            ...
        )

    def compute_heston_prob(self, sec_id: int) -> float:
        sec = self.secs.read_sec(sec_id)
        sec_ticker = sec['ticker']
        sec_label = sec['label']
        sec_url = sec['url']
        sec_expiry = pm.get_event_expiry(sec_url).date().strftime('%Y-%m-%d')

        mkt_data = hi.get_heston_inputs(sec_ticker, sec_expiry)
        strikes = mkt_data.strikes
        dK = float(strikes[1] - strikes[0])

        if mkt_data.atm_iv is not None:
            v0 = float(mkt_data.atm_iv) ** 2
        else:
            v0 = 0.04

        price_range = self._parse_price_range(sec_label)

        h_model = Heston(
            S0=mkt_data.S0,
            tau=mkt_data.tau,
            r=mkt_data.r,
            kappa=2.0,
            theta=0.04,
            v0=v0,
            rho=-0.7,
            sigma=0.30,
            lambd=0.0,
            K_min=float(strikes.min()),
            K_max=float(strikes.max()),
            dK=dK
        )
        self._breeden_litzenberger_pdf(h_model)
        norm_prob = self.prob_between(h_model, price_range[0], price_range[1], renormalize=True)

        return norm_prob

    def _breeden_litzenberger_pdf(self, model: Heston) -> pd.DataFrame:
        if model.call_prices is None or model.strikes is None:
            self._compute_call_curve(model)

        strikes = model.strikes
        call_prices = model.call_prices
        dK = model.dK

        curvature = np.full_like(call_prices, np.nan)

        for i in range(1, len(strikes) - 1):
            curvature[i] = (
                call_prices[i + 1]
                - 2.0 * call_prices[i]
                + call_prices[i - 1]
            ) / (dK**2)

        bl_pdf = np.exp(model.r * model.tau) * curvature
        bl_pdf = np.maximum(bl_pdf, 0.0)

        model.bl_pdf = bl_pdf

        return pd.DataFrame(
            {
                'strike': strikes,
                'call': call_prices,
                'bl_pdf': bl_pdf,
            }
        )

    def _compute_call_curve(self, model: Heston):
        strikes = np.arange(model.K_min, model.K_max + model.dK, model.dK)
        call_prices = np.array([model.heston_price_rec(K) for K in strikes])

        model.strikes = strikes
        model.call_prices = call_prices

        return pd.DataFrame({'strike': strikes, 'call': call_prices})
    
    def prob_between(self, model: Heston, S_low, S_high, renormalize=False):
        if model.bl_pdf is None or model.strikes is None:
            self._breeden_litzenberger_pdf(model)

        strikes = model.strikes
        bl_pdf = model.bl_pdf

        mask = (~np.isnan(bl_pdf)) & (strikes >= S_low) & (strikes <= S_high)

        if not np.any(mask):
            return 0.0

        num = np.trapezoid(bl_pdf[mask], strikes[mask])

        if not renormalize:
            return num

        mask_all = ~np.isnan(bl_pdf)
        denom = np.trapezoid(bl_pdf[mask_all], strikes[mask_all])
        if denom <= 0:
            return num
        return num / denom

    def _parse_price_range(self, label: str, def_max: int=100_000_000, def_min: int=0) -> tuple:
        s = label.replace('$', '').replace(',', '').strip()
        dashes = r'\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D'
        s_norm = re.sub(f"[{dashes}]", "-", s)

        if '-' in s_norm:
            lo_str, hi_str = s_norm.split('-', 1)
            lo = float(lo_str.strip())
            hi = float(hi_str.strip())
            return lo, hi

        if s_norm.startswith('<'):
            hi = float(s[1:].strip())
            return def_min, hi

        if s_norm.startswith('>'):
            lo = float(s[1:].strip())
            return lo, def_max

        raise ValueError(f'Unrecognized price range label: {label!r}')
