import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import yfinance as yf
import datetime as dt
from datetime import timezone

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

        today = dt.datetime.now(timezone.utc).date()
        try:
            req_expiry_date = dt.datetime.strptime(self.expiry, '%Y-%m-%d').date()
        except Exception as e:
            raise ValueError(f'Invalid expiry {self.expiry!r}, expected YYYY-MM-DD') from e

        available_expiries = list(getattr(t, 'options', []))
        if not available_expiries:
            raise ValueError(f'No listed options expiries for {self.ticker}')

        available_dates = [
            dt.datetime.strptime(e, '%Y-%m-%d').date() for e in available_expiries
        ]

        if self.expiry in available_expiries:
            chosen_date = req_expiry_date
        else:
            future_candidates = [
                d for d in available_dates
                if d >= req_expiry_date and d > today
            ]

            if not future_candidates:
                future_candidates = [d for d in available_dates if d > today]

            if not future_candidates:
                raise ValueError(f'No future expiries available for {self.ticker}')

            chosen_date = min(future_candidates)

        self.expiry_date = chosen_date
        self.expiry_str = chosen_date.strftime('%Y-%m-%d')
        self.expiry = self.expiry_str

        days_to_expiry = (self.expiry_date - today).days
        if days_to_expiry <= 0:
            raise ValueError('Expiry is not in the future')

        hist = t.history(period='1d')
        if hist.empty:
            raise RuntimeError('No price history returned from yfinance.')

        self.price = float(hist['Close'].iloc[-1])

        opt = t.option_chain(self.expiry_str)
        calls = opt.calls

        if calls.empty:
            raise RuntimeError(f'No call options for {self.ticker} on {self.expiry_str}')

        valid = calls.dropna(subset=['impliedVolatility'])
        if valid.empty:
            raise RuntimeError(f'No valid IV data for {self.ticker} on {self.expiry_str}')

        atm_idx = (valid['strike'] - self.price).abs().idxmin()
        row = valid.loc[atm_idx]

        self.atm_strike = float(row['strike'])
        self.iv_annual = float(row['impliedVolatility'])

    def _compute_params(self):
        today = dt.datetime.now(timezone.utc).date()
        T = (self.expiry_date - today).days / 365.0

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
