"""SQLite persistence for candles, trades, account state and equity curve."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    pair TEXT NOT NULL,
    granularity TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (pair, granularity, start_ts)
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT, pair TEXT, side TEXT, ts INTEGER,
    price REAL, qty REAL, fee REAL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT, pair TEXT,
    entry_ts INTEGER, entry_price REAL,
    exit_ts INTEGER, exit_price REAL,
    qty REAL, entry_fee REAL, exit_fee REAL,
    pnl REAL, pnl_pct REAL
);
CREATE TABLE IF NOT EXISTS account_state (
    key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    pair TEXT NOT NULL,
    equity REAL,
    PRIMARY KEY (ts, strategy, pair)
);
CREATE INDEX IF NOT EXISTS idx_candles ON candles (pair, granularity, start_ts);
"""


class Store:
    """Thin wrapper around a sqlite3 connection with typed helpers."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------ candles
    def upsert_candles(self, pair: str, granularity: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        rows = [
            (pair, granularity, int(ts.timestamp()), float(r.open), float(r.high), float(r.low),
             float(r.close), float(r.volume))
            for ts, r in df.iterrows()
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO candles
               (pair, granularity, start_ts, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def load_candles(self, pair: str, granularity: str,
                     start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
        sql = "SELECT start_ts, open, high, low, close, volume FROM candles WHERE pair=? AND granularity=?"
        params: List[Any] = [pair, granularity]
        if start is not None:
            sql += " AND start_ts >= ?"
            params.append(int(start))
        if end is not None:
            sql += " AND start_ts <= ?"
            params.append(int(end))
        sql += " ORDER BY start_ts"
        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["start_ts", "open", "high", "low", "close", "volume"])
        df["start"] = pd.to_datetime(df["start_ts"], unit="s", utc=True)
        return df.set_index("start").drop(columns=["start_ts"]).astype(float)

    def candle_count(self, pair: str, granularity: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM candles WHERE pair=? AND granularity=?", (pair, granularity)
        )
        return int(cur.fetchone()[0])

    # -------------------------------------------------------------- fills
    def record_fill(self, strategy: str, pair: str, side: str, ts: int,
                    price: float, qty: float, fee: float) -> None:
        self.conn.execute(
            "INSERT INTO fills (strategy, pair, side, ts, price, qty, fee) VALUES (?,?,?,?,?,?,?)",
            (strategy, pair, side, int(ts), float(price), float(qty), float(fee)),
        )
        self.conn.commit()

    # ------------------------------------------------------------- trades
    def record_trade(self, strategy: str, pair: str, entry_ts: int, entry_price: float,
                     exit_ts: int, exit_price: float, qty: float,
                     entry_fee: float, exit_fee: float, pnl: float, pnl_pct: float) -> None:
        self.conn.execute(
            """INSERT INTO trades (strategy, pair, entry_ts, entry_price, exit_ts, exit_price,
                                   qty, entry_fee, exit_fee, pnl, pnl_pct)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (strategy, pair, int(entry_ts), float(entry_price), int(exit_ts), float(exit_price),
             float(qty), float(entry_fee), float(exit_fee), float(pnl), float(pnl_pct)),
        )
        self.conn.commit()

    def load_trades(self, strategy: Optional[str] = None, pair: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []
        if strategy:
            sql += " AND strategy=?"
            params.append(strategy)
        if pair:
            sql += " AND pair=?"
            params.append(pair)
        sql += " ORDER BY exit_ts"
        df = pd.read_sql_query(sql, self.conn, params=params)
        return df

    # ------------------------------------------------------ account state
    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_state(self, key: str) -> Optional[Any]:
        cur = self.conn.execute("SELECT value FROM account_state WHERE key=?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    # --------------------------------------------------------- equity curve
    def save_equity(self, ts: int, points: Iterable[tuple[str, str, float]]) -> None:
        rows = [(int(ts), strategy, pair, float(equity)) for strategy, pair, equity in points]
        self.conn.executemany(
            "INSERT OR REPLACE INTO equity_curve (ts, strategy, pair, equity) VALUES (?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def load_equity(self, strategy: str, pair: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT ts, equity FROM equity_curve WHERE strategy=? AND pair=? ORDER BY ts",
            self.conn, params=(strategy, pair),
        )
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df.set_index("time")["equity"]
