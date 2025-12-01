import unittest
import tempfile
import sqlite3
from pathlib import Path
import esmi.secs as secs_mod
from esmi.secs import Securities

class SecuritiesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / 'secs.db'
        secs_mod.DB_PATH = self.db_path
        self.secs = Securities()

    def tearDown(self):
        self.secs.secs_db.close()
        self.tmpdir.cleanup()

    def test_read_sec_returns_no_data_when_missing(self):
        res = self.secs.read_sec(1)
        self.assertEqual(res, 'No data found for ID')

    def test_create_sec_inserts_and_returns_row(self):
        row = self.secs.create_sec(
            ticker='MSFT',
            label='$400-420',
            url='https://example.com/msft',
            pred_mkt_prob=0.6,
            opt_mkt_prob=0.5,
        )
        self.assertIsInstance(row, sqlite3.Row)
        self.assertEqual(row['id'], 1)
        self.assertEqual(row['ticker'], 'MSFT')
        self.assertEqual(row['label'], '$400-420')
        self.assertEqual(row['url'], 'https://example.com/msft')
        self.assertEqual(row['pred_mkt_prob'], 0.6)
        self.assertEqual(row['opt_mkt_prob'], 0.5)

    def test_update_sec_partial_update_keeps_other_fields(self):
        row = self.secs.create_sec(
            ticker='MSFT',
            label='$400-420',
            url='https://example.com/msft',
            pred_mkt_prob=0.6,
            opt_mkt_prob=0.5,
        )
        sec_id = row['id']

        updated = self.secs.update_sec(
            sec_id,
            ticker='AAPL',
            pred_mkt_prob=0.7,
        )

        self.assertEqual(updated['id'], sec_id)
        self.assertEqual(updated['ticker'], 'AAPL')
        self.assertEqual(updated['label'], '$400-420')
        self.assertEqual(updated['url'], 'https://example.com/msft')
        self.assertEqual(updated['pred_mkt_prob'], 0.7)
        self.assertEqual(updated['opt_mkt_prob'], 0.5)

    def test_delete_sec_returns_correct_messages(self):
        row = self.secs.create_sec(
            ticker='MSFT',
            label='$400-420',
            url='https://example.com/msft',
        )
        sec_id = row['id']

        msg = self.secs.delete_sec(sec_id)
        self.assertEqual(msg, f'Deleted security with id {sec_id}')

        msg2 = self.secs.delete_sec(sec_id)
        self.assertEqual(msg2, f'No security found with id {sec_id}')

    def test_iter_yields_all_securities(self):
        self.secs.create_sec(
            ticker='MSFT',
            label='$400-420',
            url='https://example.com/msft',
        )
        self.secs.create_sec(
            ticker='AAPL',
            label='$180-190',
            url='https://example.com/aapl',
        )

        rows = list(self.secs)
        self.assertEqual(len(rows), 2)
        tickers = [r['ticker'] for r in rows]
        self.assertEqual(tickers, ['MSFT', 'AAPL'])

    def test_str_no_securities(self):
        s = str(self.secs)
        self.assertEqual(s, 'Securities: no securities stored')

    def test_str_with_securities_expected_format(self):
        row = self.secs.create_sec(
            ticker='MSFT',
            label='$400-420',
            url='https://example.com/msft',
            pred_mkt_prob=0.6,
            opt_mkt_prob=None,
        )
        sec_id = row['id']

        s = str(self.secs)
        lines = s.splitlines()
        self.assertEqual(lines[0], 'Securities:')
        self.assertEqual(
            lines[1],
            f'[{sec_id}] MSFT "$400-420": https://example.com/msft '
            f'(pred=0.6, opt=N/A)'
        )


if __name__ == '__main__':
    unittest.main()
