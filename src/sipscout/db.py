"""SQLite storage: records, runs, and API quota tracking."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_DB
from .models import Record, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  native_id TEXT NOT NULL,
  app TEXT,
  container TEXT,
  container_title TEXT,
  parent_id TEXT,
  url TEXT NOT NULL,
  author_hash TEXT NOT NULL,
  title TEXT,
  text TEXT NOT NULL,
  rating INTEGER,
  score INTEGER,
  created_at TEXT NOT NULL,
  app_version TEXT,
  lang TEXT,
  query TEXT,
  matched_keywords TEXT,
  is_relevant INTEGER NOT NULL DEFAULT 0,
  collected_at TEXT NOT NULL,
  raw TEXT,
  extraction TEXT,
  extraction_version TEXT,
  extracted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);
CREATE INDEX IF NOT EXISTS idx_records_relevant ON records(is_relevant);

CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  stage TEXT NOT NULL,
  source TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  fetched INTEGER DEFAULT 0,
  inserted INTEGER DEFAULT 0,
  updated INTEGER DEFAULT 0,
  errors INTEGER DEFAULT 0,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
  call_id INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  stage TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT,
  records INTEGER,
  input_tokens INTEGER,
  output_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS quota (
  day TEXT NOT NULL,
  service TEXT NOT NULL,
  units INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, service)
);
"""

_COLUMNS = [
    "id", "source", "native_id", "app", "container", "container_title", "parent_id", "url",
    "author_hash", "title", "text", "rating", "score", "created_at", "app_version", "lang",
    "query", "matched_keywords", "is_relevant", "collected_at", "raw",
]
_UPDATABLE = [c for c in _COLUMNS if c not in ("id", "collected_at")]


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database (creating schema if needed)."""
    path = Path(path) if path and str(path) != ":memory:" else path
    if path is None:
        path = DEFAULT_DB
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _row(rec: Record) -> dict:
    d = rec.model_dump(mode="json")
    d["matched_keywords"] = json.dumps(d["matched_keywords"], ensure_ascii=False)
    d["raw"] = json.dumps(d["raw"], ensure_ascii=False, default=str)
    d["is_relevant"] = int(d["is_relevant"])
    return d


def upsert_records(conn: sqlite3.Connection, records: Iterable[Record]) -> tuple[int, int]:
    """Insert or update by id. Extraction columns are preserved. Returns (inserted, updated)."""
    rows = [_row(r) for r in records]
    if not rows:
        return 0, 0
    ids = [r["id"] for r in rows]
    existing: set[str] = set()
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = f"SELECT id FROM records WHERE id IN ({','.join('?' * len(chunk))})"
        existing.update(x[0] for x in conn.execute(q, chunk))
    cols = ",".join(_COLUMNS)
    params = ",".join(f":{c}" for c in _COLUMNS)
    updates = ",".join(f"{c}=excluded.{c}" for c in _UPDATABLE)
    conn.executemany(
        f"INSERT INTO records ({cols}) VALUES ({params}) ON CONFLICT(id) DO UPDATE SET {updates}",
        rows,
    )
    conn.commit()
    updated = sum(1 for i in ids if i in existing)
    return len(ids) - updated, updated


def start_run(conn: sqlite3.Connection, stage: str, source: str | None) -> int:
    """Create a runs row and return its id."""
    cur = conn.execute(
        "INSERT INTO runs (stage, source, started_at) VALUES (?,?,?)",
        (stage, source, utcnow().isoformat()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, fetched: int, inserted: int, updated: int,
               errors: int, notes: str = "") -> None:
    """Close a runs row with its counters."""
    conn.execute(
        "UPDATE runs SET finished_at=?, fetched=?, inserted=?, updated=?, errors=?, notes=? WHERE run_id=?",
        (utcnow().isoformat(), fetched, inserted, updated, errors, notes, run_id),
    )
    conn.commit()


def quota_used(conn: sqlite3.Connection, service: str, day: date | None = None) -> int:
    """Units used today for a service."""
    day = day or utcnow().date()
    row = conn.execute("SELECT units FROM quota WHERE day=? AND service=?", (day.isoformat(), service)).fetchone()
    return int(row[0]) if row else 0


def add_quota(conn: sqlite3.Connection, service: str, units: int, day: date | None = None) -> None:
    """Add units to today's usage."""
    day = (day or utcnow().date()).isoformat()
    conn.execute(
        "INSERT INTO quota (day, service, units) VALUES (?,?,?) "
        "ON CONFLICT(day, service) DO UPDATE SET units = units + excluded.units",
        (day, service, units),
    )
    conn.commit()


def count_records(conn: sqlite3.Connection) -> int:
    """Total record count."""
    return int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
