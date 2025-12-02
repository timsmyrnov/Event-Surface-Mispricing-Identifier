import re
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from esmi.secs import Securities
from esmi.black_scholes import BlackScholes
from esmi.heston import Heston
from esmi import heston_inputs as hi
from esmi import polymarket as pm

class Pricer:
    def __init__(self, secs: Securities | None=None):
        self.secs = secs or Securities()

    def compute_black_scholes_prob(self, sec_id: int) -> float:
        sec = self.secs.read_sec(sec_id)
        sec_ticker = sec['ticker']
        sec_label = sec['label']
        sec_url = sec['url']
        sec_expiry = pm.get_event_expiry(sec_url).date().strftime('%Y-%m-%d')

        bs_model = BlackScholes(sec_ticker, sec_expiry)

        price_range = self._parse_price_range(sec_label)

        return self._black_scholes_inter_prob(bs_model, price_range[0], price_range[1], renormalize=True)

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
        norm_prob = self._heston_inter_prob(h_model, price_range[0], price_range[1], renormalize=True)

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
    
    def _heston_inter_prob(self, model: Heston, S_low: float, S_high: float, renormalize: bool=False) -> float:
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
    
    def _black_scholes_inter_prob(self, model: BlackScholes, S_low: float, S_high: float, renormalize: bool=False) -> float:
        if S_high < S_low:
            S_low, S_high = S_high, S_low

        dist = lognorm(s=model.sigma, scale=np.exp(model.mu))

        p_raw = dist.cdf(S_high) - dist.cdf(S_low)

        if not renormalize:
            return float(p_raw)

        grid_low = lognorm.ppf(model.q_low, s=model.sigma, scale=np.exp(model.mu))
        grid_high = lognorm.ppf(model.q_high, s=model.sigma, scale=np.exp(model.mu))

        p_window = dist.cdf(grid_high) - dist.cdf(grid_low)
        if p_window <= 0:
            return float('nan')

        lo_clip = max(S_low, grid_low)
        hi_clip = min(S_high, grid_high)
        if hi_clip <= lo_clip:
            return 0.0

        p_raw_window = dist.cdf(hi_clip) - dist.cdf(lo_clip)
        return float(p_raw_window / p_window)

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

if __name__ == '__main__':
    p = Pricer()
    s = Securities()

    l = len(s)

    total_error = 0
    max_error = 0
    max_error_id = None

    for i in range(1, l):
        bs = p.compute_black_scholes_prob(i)
        h = p.compute_heston_prob(i)
        print(bs, h, '\n')

        curr_error = abs(bs - h)
        if curr_error > max_error:
            max_error = curr_error
            max_error_id = i

        total_error += curr_error

    print(f'Average error: {total_error / l}, Max error: {max_error}, Max error id: {max_error_id}, Tests: {l}')
