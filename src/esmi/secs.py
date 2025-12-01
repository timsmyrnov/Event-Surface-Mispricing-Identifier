import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / 'secs.db'

class Securities:
    def __init__(self):
        self.secs_db = sqlite3.connect(DB_PATH)
        self.secs_db.row_factory = sqlite3.Row
        self.db_cursor = self.secs_db.cursor()

        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS secs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            label TEXT,
            url TEXT,
            pred_mkt_prob REAL,
            opt_mkt_prob REAL
            )
        ''')
        self.secs_db.commit()

    def create_sec(
        self,
        ticker: str,
        label: str,
        url: str,
        pred_mkt_prob: float=None,
        opt_mkt_prob: float=None
    ) -> sqlite3.Row | str:
        self.db_cursor.execute('''
            INSERT INTO secs
            (ticker, label, url, pred_mkt_prob, opt_mkt_prob)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (ticker, label, url, pred_mkt_prob, opt_mkt_prob),
        )
        self.secs_db.commit()

        return self.read_sec(self.db_cursor.lastrowid)

    def read_sec(self, id: int) -> sqlite3.Row | str:
        self.db_cursor.execute('SELECT * FROM secs WHERE id = ?', (id,))
        row = self.db_cursor.fetchone()

        return row if row else 'No data found for ID'

    def update_sec(
        self,
        id: int,
        ticker: str=None,
        label: str=None,
        url: str=None,
        pred_mkt_prob: float=None,
        opt_mkt_prob: float=None
    ) -> sqlite3.Row | str:
        self.db_cursor.execute('''
            UPDATE secs
            SET
            ticker = COALESCE(?, ticker),
            label = COALESCE(?, label),
            url = COALESCE(?, url),
            pred_mkt_prob = COALESCE(?, pred_mkt_prob),
            opt_mkt_prob = COALESCE(?, opt_mkt_prob)
            WHERE id = ?
            ''',
            (ticker, label, url, pred_mkt_prob, opt_mkt_prob, id),
        )
        self.secs_db.commit()
        return self.read_sec(id)

    def delete_sec(self, id: int) -> str:
        self.db_cursor.execute('DELETE FROM secs WHERE id = ?', (id,))
        self.secs_db.commit()

        return f'No security found with id {id}' if self.db_cursor.rowcount == 0 else f'Deleted security with id {id}'
    
    def __iter__(self):
        cur = self.secs_db.cursor()
        cur.execute(
            'SELECT id, ticker, label, url, pred_mkt_prob, opt_mkt_prob FROM secs'
        )
        rows = cur.fetchall()
        for row in rows:
            yield row
    
    def __str__(self) -> str:
        self.db_cursor.execute(
            'SELECT id, ticker, label, url, pred_mkt_prob, opt_mkt_prob FROM secs'
        )
        rows = self.db_cursor.fetchall()
        if not rows:
            return 'Securities: no securities stored'

        lines = ['Securities:']
        for sec_id, ticker, label, url, pred_price, opt_price in rows:
            lines.append(
                f'[{sec_id}] {ticker} "{label}": {url} '
                f'(pred={pred_price if pred_price is not None else 'N/A'}, '
                f'opt={opt_price if opt_price is not None else 'N/A'})'
            )

        return '\n'.join(lines)
