"""
SQLite storage (WAL mode).

v2 used DuckDB, which holds an exclusive file lock on Windows: nothing else
can even *read* the database while the monitor runs. SQLite in WAL mode lets
`monitor.py status` / `report` read concurrently, ships with Python, and is
plenty fast for one row per second.

Tables
  samples  1 Hz raw telemetry            (short retention, forensic detail)
  windows  10 s aggregates + ML score    (the training set)
  events   detected incidents            (long retention)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, WINDOW_COLUMNS
from .sampler import SAMPLE_COLUMNS

SCHEMA_VERSION = 3
WINDOW_EXTRA = ["score", "anomalous", "top_procs", "fg_name"]


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=10)
    else:
        conn = sqlite3.connect(str(path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _clean(v):
    """NaN → NULL so SQLite stores 'unknown' rather than a float that breaks AVG()."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


class Storage:
    def __init__(self, path: Path):
        self.path = path
        self.conn = connect(path)
        self._create_schema()
        self._samples: list[tuple] = []
        self._windows: list[tuple] = []

    def _create_schema(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row and int(row[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path.name} has schema v{row[0]}, expected v{SCHEMA_VERSION}. "
                f"Run `python monitor.py reset` to start fresh.")
        sample_cols = ", ".join(f"{k} REAL" for k in SAMPLE_COLUMNS[1:])
        window_cols = ", ".join(f"{k} REAL" for k in WINDOW_COLUMNS[1:])
        with c:
            c.execute(f"CREATE TABLE IF NOT EXISTS samples (ts REAL PRIMARY KEY, {sample_cols}) WITHOUT ROWID")
            c.execute(f"""CREATE TABLE IF NOT EXISTS windows (
                ts REAL PRIMARY KEY, {window_cols},
                score REAL, anomalous INTEGER, top_procs TEXT, fg_name TEXT) WITHOUT ROWID""")
            c.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, severity TEXT NOT NULL,
                ts_start REAL NOT NULL, ts_end REAL, is_open INTEGER NOT NULL,
                title TEXT NOT NULL, details TEXT)""")
            c.execute("CREATE INDEX IF NOT EXISTS events_by_start ON events(ts_start)")
            c.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    # ── Buffered telemetry writes ─────────────────────────────────────

    def add_sample(self, s: dict) -> None:
        self._samples.append(tuple(_clean(s[k]) for k in SAMPLE_COLUMNS))

    def add_window(self, w: dict) -> None:
        extra = (_clean(w.get("score")), w.get("anomalous"),
                 json.dumps(w["top_procs"]) if w.get("top_procs") else None, w.get("fg_name"))
        self._windows.append(tuple(_clean(w[k]) for k in WINDOW_COLUMNS) + extra)

    def flush(self) -> None:
        if not self._samples and not self._windows:
            return
        with self.conn:
            if self._samples:
                ph = ", ".join("?" * len(SAMPLE_COLUMNS))
                self.conn.executemany(f"INSERT OR REPLACE INTO samples VALUES ({ph})", self._samples)
            if self._windows:
                ph = ", ".join("?" * (len(WINDOW_COLUMNS) + len(WINDOW_EXTRA)))
                self.conn.executemany(f"INSERT OR REPLACE INTO windows VALUES ({ph})", self._windows)
        self._samples.clear()
        self._windows.clear()

    # ── Events ────────────────────────────────────────────────────────

    def insert_event(self, kind: str, severity: str, ts_start: float, title: str, details: dict) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO events (kind, severity, ts_start, ts_end, is_open, title, details) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (kind, severity, ts_start, ts_start, title, json.dumps(details, default=str)))
        return cur.lastrowid

    def update_event(self, event_id: int, *, severity: str, ts_end: float, is_open: bool,
                     title: str, details: dict) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE events SET severity=?, ts_end=?, is_open=?, title=?, details=? WHERE id=?",
                (severity, ts_end, int(is_open), title, json.dumps(details, default=str), event_id))

    def recent_events(self, kinds: list[str], since: float) -> list[dict]:
        rows = self.conn.execute(
            f"SELECT kind, ts_start, details FROM events "
            f"WHERE ts_start >= ? AND kind IN ({', '.join('?' * len(kinds))})",
            [since, *kinds]).fetchall()
        return [{"kind": k, "ts": ts, "details": json.loads(d) if d else {}} for k, ts, d in rows]

    def close_orphaned_events(self) -> int:
        """Events left open by a crash or power loss: close them at their last update."""
        with self.conn:
            return self.conn.execute("UPDATE events SET is_open=0 WHERE is_open=1").rowcount

    # ── Maintenance ───────────────────────────────────────────────────

    def prune(self, now: float, raw_hours: float, window_days: float, event_days: float) -> int:
        with self.conn:
            n = self.conn.execute("DELETE FROM samples WHERE ts < ?", (now - raw_hours * 3600,)).rowcount
            n += self.conn.execute("DELETE FROM windows WHERE ts < ?", (now - window_days * 86400,)).rowcount
            n += self.conn.execute("DELETE FROM events WHERE ts_start < ? AND is_open = 0",
                                   (now - event_days * 86400,)).rowcount
        return n

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.conn.close()


# ── Read-side helpers (safe to call from other threads / processes) ──

def load_training_windows(path: Path, since: float, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Random sample of recent windows → (timestamps, (n, n_features) array in display units)."""
    conn = connect(path, readonly=True)
    try:
        cols = ", ".join(["ts"] + FEATURE_NAMES)
        rows = conn.execute(
            f"SELECT {cols} FROM windows WHERE ts >= ? ORDER BY random() LIMIT ?",
            (since, limit)).fetchall()
    finally:
        conn.close()
    a = np.array(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES) + 1)
    return a[:, 0], a[:, 1:]


def query_events(path: Path, since: float, kinds: list[str] | None = None) -> list[dict]:
    conn = connect(path, readonly=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM events WHERE (ts_end >= ? OR is_open = 1)"
        args: list = [since]
        if kinds:
            sql += f" AND kind IN ({', '.join('?' * len(kinds))})"
            args += kinds
        rows = conn.execute(sql + " ORDER BY ts_start", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d["details"]) if d["details"] else {}
        out.append(d)
    return out


def db_summary(path: Path) -> dict:
    conn = connect(path, readonly=True)
    conn.row_factory = sqlite3.Row
    try:
        out = {}
        for table in ("samples", "windows"):
            r = conn.execute(f"SELECT COUNT(*) AS n, MIN(ts) AS first, MAX(ts) AS last FROM {table}").fetchone()
            out[table] = dict(r)
        out["events"] = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        last = conn.execute("SELECT * FROM samples ORDER BY ts DESC LIMIT 1").fetchone()
        out["latest_sample"] = dict(last) if last else None
        out["size_mb"] = round(sum(p.stat().st_size for p in path.parent.glob(path.name + "*")) / 2**20, 1)
        out["now"] = time.time()
        return out
    finally:
        conn.close()


def load_windows(path: Path, since: float) -> list[dict]:
    """All windows since `since`, oldest first (for offline replay)."""
    conn = connect(path, readonly=True)
    try:
        cols = ["ts"] + FEATURE_NAMES
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM windows WHERE ts >= ? ORDER BY ts",
                            (since,)).fetchall()
    finally:
        conn.close()
    nan = float("nan")
    return [{c: (nan if v is None else v) for c, v in zip(cols, r)} for r in rows]
