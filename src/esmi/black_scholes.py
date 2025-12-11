import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import yfinance as yf
from datetime import datetime as dt, timezone
import esmi.market_data as md

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
        self.ann_iv = None
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
        self.expiry = md.get_closest_expiry()
        self.atm_strike, self.ann_iv = md.get_atm_data(self.ticker, self.expiry)
        self.price = md.get_latest_close_price(self.ticker)

    def _compute_params(self):
        today = dt.now(timezone.utc).date()
        expiry_date = dt.strptime(self.expiry, '%Y-%m-%d').date()
        T = (expiry_date - today).days / 365.0

        if T <= 0:
            raise ValueError('Expiry must be in the future to build a terminal PDF.')

        self.T = T
        self.iv = self.ann_iv * np.sqrt(T)

        self.mu = np.log(self.price) - 0.5 * self.iv ** 2
        self.sigma = self.iv

        self.mean = float(np.exp(self.mu + 0.5 * self.sigma ** 2))
        self.dte = int(round(self.T * 365))

    def _build_grid(self):
        self.xs = np.linspace(
            lognorm.ppf(self.q_low, s=self.sigma, scale=np.exp(self.mu)),
            lognorm.ppf(self.q_high, s=self.sigma, scale=np.exp(self.mu)),
            self.num_points,
        )
        self.pdf = lognorm.pdf(self.xs, s=self.sigma, scale=np.exp(self.mu))

    def plot_pdf(
        self,
        ax: plt.Axes | None=None,
        label: str='current vol',
        color: str='blue',
        show: bool=True,
    ) -> plt.Axes:

        if ax is None:
            _, ax = plt.subplots()

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
