"""SQLite persistence: trades, positions, equity snapshots, signal log, meta.

Production notes
  - WAL journal mode + busy timeout for concurrent CLI + bot access.
  - Indexes on hot query paths (mode/status, timestamps).
  - All writes are transactional; connections are short-lived and pooled
    behind a thread lock (safe for the single-process trader loop).
  - :meth:`Store.backup` produces a consistent snapshot file for ops.
"""

from __future__ import annotations

import contextlib
import csv
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    quote_amt REAL NOT NULL,
    fee REAL DEFAULT 0,
    reason TEXT DEFAULT '',
    exchange_order_id TEXT,
    raw TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    opened_ts TEXT,
    entry_price REAL NOT NULL,
    qty REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    trail_high REAL,
    status TEXT NOT NULL DEFAULT 'open',
    closed_ts TEXT,
    exit_price REAL,
    pnl_quote REAL,
    mode TEXT NOT NULL DEFAULT 'paper',
    stop_order_id TEXT,
    take_order_id TEXT,
    exchange_stop_price REAL,
    UNIQUE(symbol, mode, status)
);
CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_pos_mode_status ON positions(mode, status);
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    mode TEXT NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_mode ON equity(mode, id);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    reason TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(id DESC);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    """Thread-safe SQLite journal. ``path`` may be ``:memory:``-style for tests."""

    def __init__(self, path: str):
        self.path = path
        self._is_memory = path in (":memory:", ":memory-test:") or path.startswith("file::memory:")
        if not self._is_memory and not path.startswith("file:"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mem_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._mem_conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        with self._conn() as c:
            c.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent schema upgrades for pre-existing journals."""
        for col in ("stop_order_id", "take_order_id", "exchange_stop_price"):
            coldef = f"{col} TEXT" if col != "exchange_stop_price" else f"{col} REAL"
            try:
                with self._conn() as c:
                    c.execute(f"ALTER TABLE positions ADD COLUMN {coldef}")
            except sqlite3.DatabaseError:
                pass  # column already present

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._is_memory and self._mem_conn is not None:
                try:
                    yield self._mem_conn
                    self._mem_conn.commit()
                except Exception:
                    with contextlib.suppress(Exception):
                        self._mem_conn.rollback()
                    raise
                return
            c = sqlite3.connect(self.path, timeout=30.0)
            c.row_factory = sqlite3.Row
            try:
                try:
                    c.execute("PRAGMA journal_mode=WAL")
                    c.execute("PRAGMA busy_timeout=30000")
                    c.execute("PRAGMA synchronous=NORMAL")
                except sqlite3.DatabaseError:
                    pass
                yield c
                c.commit()
            finally:
                c.close()

    # --------------- meta ---------------
    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    # --------------- trades ---------------
    def insert_trade(self, t: dict[str, Any]) -> int:
        cols = (
            "ts",
            "mode",
            "symbol",
            "side",
            "qty",
            "price",
            "quote_amt",
            "fee",
            "reason",
            "exchange_order_id",
            "raw",
        )
        vals = [t.get(c) for c in cols]
        with self._conn() as c:
            cur = c.execute(
                f"INSERT INTO trades({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                vals,
            )
            return int(cur.lastrowid)

    def trades(
        self,
        mode: str | None = None,
        limit: int = 100,
        symbol: str | None = None,
        side: str | None = None,
    ) -> list[dict]:
        q = "SELECT * FROM trades"
        clauses: list[str] = []
        args: list = []
        if mode:
            clauses.append("mode=?")
            args.append(mode)
        if symbol:
            clauses.append("symbol=?")
            args.append(symbol)
        if side:
            clauses.append("side=?")
            args.append(side)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def export_trades_csv(self, path: str, mode: str | None = None) -> int:
        rows = self.trades(mode=mode, limit=100_000)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "ts",
                    "mode",
                    "symbol",
                    "side",
                    "qty",
                    "price",
                    "quote_amt",
                    "fee",
                    "reason",
                    "exchange_order_id",
                ],
            )
            w.writeheader()
            for r in reversed(rows):  # chronological for spreadsheets
                w.writerow({k: r.get(k) for k in w.fieldnames})
        return len(rows)

    # --------------- positions ---------------
    def open_position(self, p: dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO positions(symbol,strategy,opened_ts,entry_price,qty,stop_loss,"
                "take_profit,trail_high,status,mode) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    p["symbol"],
                    p["strategy"],
                    utcnow(),
                    p["entry_price"],
                    p["qty"],
                    p.get("stop_loss"),
                    p.get("take_profit"),
                    p.get("entry_price"),
                    "open",
                    p["mode"],
                ),
            )
            return int(cur.lastrowid)

    def update_protection_orders(
        self,
        pos_id: int,
        stop_order_id: int | None,
        take_order_id: int | None,
        stop_trigger: float | None = None,
    ) -> None:
        """Persist exchange-native stop/TP order ids (+ the armed trigger price)."""
        with self._conn() as c:
            if stop_trigger is not None:
                c.execute(
                    "UPDATE positions SET stop_order_id=?, take_order_id=?, "
                    "exchange_stop_price=? WHERE id=?",
                    (
                        str(stop_order_id) if stop_order_id else None,
                        str(take_order_id) if take_order_id else None,
                        float(stop_trigger),
                        pos_id,
                    ),
                )
            else:
                c.execute(
                    "UPDATE positions SET stop_order_id=?, take_order_id=? WHERE id=?",
                    (
                        str(stop_order_id) if stop_order_id else None,
                        str(take_order_id) if take_order_id else None,
                        pos_id,
                    ),
                )

    def close_position(self, pos_id: int, exit_price: float, pnl_quote: float) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE positions SET status='closed', closed_ts=?, exit_price=?, pnl_quote=? "
                "WHERE id=?",
                (utcnow(), exit_price, pnl_quote, pos_id),
            )

    def get_open_position(self, symbol: str, mode: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM positions WHERE symbol=? AND mode=? AND status='open'",
                (symbol, mode),
            ).fetchone()
            return dict(row) if row else None

    def open_positions(self, mode: str) -> list[dict]:
        with self._conn() as c:
            return [
                dict(r)
                for r in c.execute(
                    "SELECT * FROM positions WHERE mode=? AND status='open' ORDER BY id", (mode,)
                )
            ]

    def update_trail(self, pos_id: int, trail_high: float, stop_loss: float | None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE positions SET trail_high=?, stop_loss=? WHERE id=?",
                (trail_high, stop_loss, pos_id),
            )

    def closed_positions(
        self, mode: str, symbol: str | None = None, limit: int = 10_000
    ) -> list[dict]:
        q = "SELECT * FROM positions WHERE mode=? AND status='closed'"
        args: list = [mode]
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    # --------------- equity ---------------
    def snapshot_equity(self, mode: str, cash: float, pos_value: float) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO equity(ts,mode,cash,positions_value,total) VALUES(?,?,?,?,?)",
                (utcnow(), mode, cash, pos_value, cash + pos_value),
            )

    def equity_history(self, mode: str, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM equity WHERE mode=? ORDER BY id DESC LIMIT ?", (mode, limit)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # --------------- signals ---------------
    def log_signal(self, sig) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO signals(ts,symbol,strategy,side,price,reason) VALUES(?,?,?,?,?,?)",
                (utcnow(), sig.symbol, sig.strategy, sig.side, sig.price, sig.reason),
            )

    def recent_signals(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            return [
                dict(r)
                for r in c.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
            ]

    # --------------- paper state ---------------
    def set_paper_state(self, cash: float) -> None:
        self.set_meta("paper_cash", repr(float(cash)))

    def get_paper_state(self, default: float) -> float:
        v = self.get_meta("paper_cash")
        try:
            return float(v) if v is not None else float(default)
        except (ValueError, TypeError):
            return float(default)

    # --------------- ops ---------------
    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            out = {}
            for tbl in ("trades", "positions", "equity", "signals"):
                try:
                    out[tbl] = c.execute(f"SELECT COUNT(*) AS n FROM {tbl}").fetchone()["n"]
                except sqlite3.DatabaseError:
                    out[tbl] = 0
            return out

    def backup(self, dest: str) -> str:
        """Consistent file backup (SQLite backup API). Returns dest path."""
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if self._is_memory:
            raise ValueError("cannot backup an in-memory database")
        src = sqlite3.connect(self.path, timeout=30.0)
        try:
            dst = sqlite3.connect(dest, timeout=30.0)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        # Fallback copy for WAL sidecars is unnecessary; backup API is consistent.
        _ = shutil  # keep import used for future file-level fallback
        return dest
