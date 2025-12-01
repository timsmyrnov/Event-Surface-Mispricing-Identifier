import unittest
import tempfile
import sqlite3
from pathlib import Path
import esmi.portfolio as pf

class PortfolioTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / 'positions.db'

        pf.DB_PATH = self.db_path

        self.portfolio = pf.Portfolio()

    def tearDown(self):
        self.portfolio.positions_db.close()
        self.tmpdir.cleanup()

    def _make_sec_row(self, sec_id=42, ticker='MSFT', label='$400-420'):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            'CREATE TABLE secs (id INTEGER, ticker TEXT, label TEXT)'
        )
        cur.execute(
            'INSERT INTO secs (id, ticker, label) VALUES (?, ?, ?)',
            (sec_id, ticker, label),
        )
        conn.commit()
        row = cur.execute('SELECT * FROM secs WHERE id = ?', (sec_id,)).fetchone()
        conn.close()
        return row

    def test_read_pos_returns_no_data_when_missing(self):
        res = self.portfolio.read_pos(1)
        self.assertEqual(res, 'No data found for ID')

    def test_create_pos_inserts_and_returns_row(self):
        sec = self._make_sec_row(sec_id=42, ticker='MSFT', label='$400-420')
        edge = (0.05, 'Yes')
        price = 0.55
        qty = 10.0

        row = self.portfolio.create_pos(sec=sec, edge=edge, price=price, qty=qty)
        self.assertIsInstance(row, sqlite3.Row)

        self.assertEqual(row['sec_id'], 42)
        self.assertEqual(row['ticker'], 'MSFT')
        self.assertEqual(row['label'], '$400-420')
        self.assertEqual(row['side'], 'Yes')
        self.assertEqual(row['vwap'], price)
        self.assertEqual(row['qty'], qty)
        self.assertEqual(row['id'], 1)

    def test_update_pos_partial_update_keeps_other_fields(self):
        sec = self._make_sec_row(sec_id=42, ticker='MSFT', label='$400-420')
        edge = (0.05, 'Yes')
        price = 0.55
        qty = 10.0

        row = self.portfolio.create_pos(sec=sec, edge=edge, price=price, qty=qty)
        pos_id = row['id']

        updated = self.portfolio.update_pos(
            pos_id,
            ticker='AAPL',
            vwap=0.60,
        )

        self.assertEqual(updated['id'], pos_id)
        self.assertEqual(updated['sec_id'], 42)
        self.assertEqual(updated['ticker'], 'AAPL')
        self.assertEqual(updated['label'], '$400-420')
        self.assertEqual(updated['side'], 'Yes')
        self.assertEqual(updated['qty'], qty)
        self.assertEqual(updated['vwap'], 0.60)

    def test_delete_pos_returns_correct_messages(self):
        sec = self._make_sec_row(sec_id=10, ticker='MSFT', label='$400-420')
        edge = (0.05, 'Yes')
        row = self.portfolio.create_pos(sec=sec, edge=edge, price=0.5, qty=5.0)
        pos_id = row['id']

        msg = self.portfolio.delete_pos(pos_id)
        self.assertEqual(msg, f'Deleted position with id {pos_id}')

        msg2 = self.portfolio.delete_pos(pos_id)
        self.assertEqual(msg2, f'No position found with id {pos_id}')

    def test_iter_yields_all_positions(self):
        sec1 = self._make_sec_row(sec_id=1, ticker='MSFT', label='$400-420')
        sec2 = self._make_sec_row(sec_id=2, ticker='AAPL', label='$180-190')

        self.portfolio.create_pos(sec1, edge=(0.05, 'Yes'), price=0.5, qty=10)
        self.portfolio.create_pos(sec2, edge=(0.03, 'No'), price=0.4, qty=20)

        positions = list(self.portfolio)
        self.assertEqual(len(positions), 2)

        tickers = [p['ticker'] for p in positions]
        self.assertEqual(tickers, ['MSFT', 'AAPL'])

    def test_str_empty_portfolio(self):
        s = str(self.portfolio)
        self.assertEqual(s, 'Portfolio: no positions stored')

    def test_str_with_positions_has_expected_format(self):
        sec = self._make_sec_row(sec_id=42, ticker='MSFT', label='$400-420')
        edge = (0.05, 'Yes')
        self.portfolio.create_pos(sec=sec, edge=edge, price=0.55, qty=10.0)

        s = str(self.portfolio)
        lines = s.splitlines()
        self.assertEqual(lines[0], 'Portfolio:')
        self.assertEqual(
            lines[1],
            '[1] MSFT Yes qty=10.0 @ 0.55 (sec_id=42) \'$400-420\'',
        )


if __name__ == '__main__':
    unittest.main()
