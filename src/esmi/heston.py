import numpy as np

class Heston:
    def __init__(
        self,
        S0,
        tau,
        r,
        kappa,
        theta,
        v0,
        rho,
        sigma,
        lambd,
        K_min=60.0,
        K_max=180.0,
        dK=1.0,
        umax=100.0,
        N=650,
    ):
        self.S0 = S0
        self.tau = tau
        self.r = r

        self.kappa = kappa
        self.theta = theta
        self.v0 = v0
        self.rho = rho
        self.sigma = sigma
        self.lambd = lambd

        self.K_min = K_min
        self.K_max = K_max
        self.dK = dK
        self.umax = umax
        self.N = N

        self.strikes = None
        self.call_prices = None
        self.bl_pdf = None

    def heston_charfunc(self, phi):
        S0 = self.S0
        v0 = self.v0
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        lambd = self.lambd
        tau = self.tau
        r = self.r

        a = kappa * theta
        b = kappa + lambd

        rspi = rho * sigma * phi * 1j

        d = np.sqrt((rspi - b) ** 2 + (phi * 1j + phi**2) * sigma**2)
        g = (b - rspi + d) / (b - rspi - d)

        exp1 = np.exp(r * phi * 1j * tau)
        term2 = S0 ** (phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g)) ** (-2 * a / sigma**2)
        exp2 = np.exp(
            a * tau * (b - rspi + d) / sigma**2
            + v0 * (b - rspi + d) * (1 - np.exp(d * tau))
            / ((1 - g * np.exp(d * tau)) * sigma**2)
        )

        return exp1 * term2 * exp2

    def heston_price_rec(self, K):
        P = 0.0
        dphi = self.umax / self.N

        for j in range(1, self.N):
            phi = dphi * (2 * j + 1) / 2.0
            numerator = self.heston_charfunc(phi - 1j) - K * self.heston_charfunc(phi)
            denominator = 1j * phi * K ** (1j * phi)
            P += dphi * numerator / denominator

        return np.real(
            (self.S0 - K * np.exp(-self.r * self.tau)) / 2.0 + P / np.pi
        )
