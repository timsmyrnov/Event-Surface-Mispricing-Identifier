import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import yfinance as yf
from datetime import datetime, timezone

class BlackScholes:
    def __init__(
        self,
        ticker: str,
        expiry: str,
        q_low: float=0.001,
        q_high: float=0.999,
        num_points: int=600,
    ):
        self.ticker = ticker
        self.expiry = expiry
        self.q_low = q_low
        self.q_high = q_high
        self.num_points = num_points

        self.price = None
        self.atm_strike = None
        self.iv_annual = None
        self.T = None
        self.iv = None
        self.mu = None
        self.sigma = None
        self.mean = None
        self.dte = None
        self.xs = None
        self.pdf = None

        self._fetch_market_data()
        self._compute_params()
        self._build_grid()

    def _fetch_market_data(self):
        t = yf.Ticker(self.ticker)

        if self.expiry not in t.options:
            raise ValueError(
                f'Expiry {self.expiry} not found. Available expiries: {t.options}'
            )

        hist = t.history(period='1d')
        if hist.empty:
            raise RuntimeError('No price history returned from yfinance.')

        self.price = float(hist['Close'].iloc[-1])

        opt = t.option_chain(self.expiry)
        calls = opt.calls

        atm_idx = (calls['strike'] - self.price).abs().idxmin()
        self.atm_strike = float(calls.loc[atm_idx, 'strike'])
        self.iv_annual = float(
            calls.loc[calls['strike'] == self.atm_strike, 'impliedVolatility'].iloc[0]
        )

    def _compute_params(self):
        now_utc = datetime.now(timezone.utc)
        expiry_dt = datetime.strptime(self.expiry, '%Y-%m-%d').replace(
            tzinfo=timezone.utc
        )
        T = (expiry_dt - now_utc).days / 365

        if T <= 0:
            raise ValueError('Expiry must be in the future to build a terminal PDF.')

        self.T = T
        self.iv = self.iv_annual * np.sqrt(T)

        self.mu = np.log(self.price) - 0.5 * self.iv**2
        self.sigma = self.iv

        self.mean = float(np.exp(self.mu + 0.5 * self.sigma**2))
        self.dte = int(round(self.T * 365))

    def _build_grid(self):
        self.xs = np.linspace(
            lognorm.ppf(self.q_low, s=self.sigma, scale=np.exp(self.mu)),
            lognorm.ppf(self.q_high, s=self.sigma, scale=np.exp(self.mu)),
            self.num_points,
        )
        self.pdf = lognorm.pdf(self.xs, s=self.sigma, scale=np.exp(self.mu))

    def prob_between(self, K_low, K_high, renormalize: bool = False) -> float:
        if K_high < K_low:
            K_low, K_high = K_high, K_low

        dist = lognorm(s=self.sigma, scale=np.exp(self.mu))

        p_raw = dist.cdf(K_high) - dist.cdf(K_low)

        if not renormalize:
            return float(p_raw)

        grid_low = lognorm.ppf(self.q_low, s=self.sigma, scale=np.exp(self.mu))
        grid_high = lognorm.ppf(self.q_high, s=self.sigma, scale=np.exp(self.mu))

        p_window = dist.cdf(grid_high) - dist.cdf(grid_low)
        if p_window <= 0:
            return float('nan')

        lo_clip = max(K_low, grid_low)
        hi_clip = min(K_high, grid_high)
        if hi_clip <= lo_clip:
            return 0.0

        p_raw_window = dist.cdf(hi_clip) - dist.cdf(lo_clip)
        return float(p_raw_window / p_window)

    def plot_pdf(
        self,
        ax: plt.Axes | None = None,
        label: str = 'current vol',
        color: str = 'blue',
        show: bool = True,
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.xs, self.pdf, color=color, linewidth=2, label=label)
        ax.axvline(self.mean, color='gray', linestyle='--', linewidth=1.5, label='mean')

        ax.set_title(
            f'Lognormal Distribution of {self.ticker} Terminal Price ({self.dte} DTE)'
        )
        ax.set_xlabel(r'$S_T$')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        if show:
            plt.show()

        return ax
