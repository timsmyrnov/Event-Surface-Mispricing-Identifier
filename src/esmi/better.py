import sqlite3
from secs import Securities
from portfolio import Portfolio
import polymarket as pm

class Better:
    def __init__(self):
        self.secs = Securities()
        self.portfolio = Portfolio()

    def try_bet(self, sec_id: int, min_edge: float=0.05) -> bool | str:
        edge = self._compute_edge(sec_id, min_edge)

        if edge is not None:
            return self.create_bet(sec_id, edge)

        return 'Bet failed'

    def create_bet(self, sec_id: int, edge: tuple) -> sqlite3.Row | str:
        sec = self.secs.read_sec(sec_id)
        sec_url = sec['url']
        sec_label = sec['label']
        _, side = edge

        shares, _, vwap = pm.max_invest_for_side_at_price(sec_url, side, sec_label)

        return self.portfolio.create_pos(sec, edge, vwap, shares)

    def _compute_edge(self, sec_id: int, min_edge: float=0.05) -> tuple | None:
        sec = self.secs.read_sec(sec_id)
        opt_mkt_price = sec['opt_mkt_prob']
        pred_mkt_price = sec['pred_mkt_prob']

        edge = opt_mkt_price - pred_mkt_price
        side = 'Yes' if edge > 0 else 'No'

        return (abs(edge), side) if abs(edge) >= min_edge else None
