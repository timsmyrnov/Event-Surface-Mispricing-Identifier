import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / 'positions.db'

class Portfolio:
    def __init__(self):
        self.positions_db = sqlite3.connect(DB_PATH)
        self.positions_db.row_factory = sqlite3.Row
        self.db_cursor = self.positions_db.cursor()

        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sec_id INTEGER,
            ticker TEXT,
            label TEXT,
            side TEXT,
            qty REAL,
            vwap REAL
            )
        ''')
        self.positions_db.commit()

    def create_pos(self, sec: sqlite3.Row, edge: tuple, price: float, qty: float) -> sqlite3.Row | str:
        sec_id = sec['id']
        sec_ticker = sec['ticker']
        sec_label = sec['label']
        side = edge[1]

        self.db_cursor.execute('''
            INSERT INTO positions
            (sec_id, ticker, label, side, vwap, qty)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (sec_id, sec_ticker, sec_label, side, price, qty),
        )
        self.positions_db.commit()

        return self.read_pos(self.db_cursor.lastrowid)
    
    def read_pos(self, id: int) -> sqlite3.Row | str:
        self.db_cursor.execute('SELECT * FROM positions WHERE id = ?', (id,))
        row = self.db_cursor.fetchone()

        return row if row else 'No data found for ID'
    
    def update_pos(
        self,
        id: int,
        sec_id: int=None,
        ticker: str=None,
        label: str=None,
        side: str=None,
        vwap: float=None,
        qty: float=None
    ) -> sqlite3.Row | str:
        self.db_cursor.execute('''
            UPDATE positions
            SET
            sec_id = COALESCE(?, sec_id),
            ticker = COALESCE(?, ticker),
            label = COALESCE(?, label),
            side = COALESCE(?, side),
            vwap = COALESCE(?, vwap),
            qty = COALESCE(?, qty)
            WHERE id = ?
            ''',
            (sec_id, ticker, label, side, vwap, qty, id),
        )
        self.positions_db.commit()
        return self.read_pos(id)
    
    def delete_pos(self, id: int) -> str:
        self.db_cursor.execute('DELETE FROM positions WHERE id = ?', (id,))
        self.positions_db.commit()

        return f'No position found with id {id}' if self.db_cursor.rowcount == 0 else f'Deleted position with id {id}'
    
    def __iter__(self):
        cur = self.positions_db.cursor()
        cur.execute(
            'SELECT id, sec_id, ticker, label, side, vwap, qty FROM positions'
        )
        rows = cur.fetchall()
        for row in rows:
            yield row

    def __len__(self) -> int:
        self.db_cursor.execute('SELECT COUNT(*) FROM portfolio')
        (count,) = self.db_cursor.fetchone()
        return count

    def __str__(self) -> str:
        self.db_cursor.execute(
            'SELECT id, sec_id, ticker, label, side, vwap, qty FROM positions'
        )
        rows = self.db_cursor.fetchall()
        if not rows:
            return 'Portfolio: no positions stored'

        lines = ['Portfolio:']
        for pos_id, sec_id, ticker, label, side, vwap, qty in rows:
            vwap_str = vwap if vwap is not None else 'N/A'
            qty_str = qty if qty is not None else 'N/A'
            lines.append(
                f'[{pos_id}] {ticker} {side} qty={qty_str} @ {vwap_str} '
                f'(sec_id={sec_id}) \'{label}\''
            )

        return '\n'.join(lines)
