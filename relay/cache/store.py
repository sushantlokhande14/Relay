from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# This is the source of truth. proxima is only a rebuildable index over the
# embeddings kept here, because it can't delete or update in place. Everything
# real (prompt, embedding, response, tokens, timestamps) lives in this table.

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt_hash       TEXT NOT NULL,
  route             TEXT NOT NULL,
  prompt_text       TEXT NOT NULL,
  embedding         BLOB,
  response_text     TEXT NOT NULL,
  model             TEXT NOT NULL,
  params_json       TEXT NOT NULL,
  prompt_tokens     INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  created_at        REAL NOT NULL,
  last_used_at      REAL NOT NULL,
  hit_count         INTEGER NOT NULL DEFAULT 0,
  alive             INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_hash  ON cache_entries(prompt_hash, route, alive);
CREATE INDEX IF NOT EXISTS idx_alive ON cache_entries(route, alive);
"""


@dataclass
class Entry:
    prompt_hash: str
    route: str
    prompt_text: str
    response_text: str
    model: str
    params_json: str
    prompt_tokens: int
    completion_tokens: int
    embedding: np.ndarray | None = None


def pack(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._db.executescript(SCHEMA)
        self._db.commit()

    def get_by_hash(self, prompt_hash: str, route: str) -> sqlite3.Row | None:
        cur = self._db.execute(
            "SELECT * FROM cache_entries WHERE prompt_hash=? AND route=? AND alive=1 LIMIT 1",
            (prompt_hash, route),
        )
        return cur.fetchone()

    def get_by_id(self, row_id: int) -> sqlite3.Row | None:
        cur = self._db.execute(
            "SELECT * FROM cache_entries WHERE id=? AND alive=1", (row_id,)
        )
        return cur.fetchone()

    def insert(self, e: Entry) -> int:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                """INSERT INTO cache_entries
                   (prompt_hash, route, prompt_text, embedding, response_text, model,
                    params_json, prompt_tokens, completion_tokens, created_at, last_used_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.prompt_hash,
                    e.route,
                    e.prompt_text,
                    pack(e.embedding),
                    e.response_text,
                    e.model,
                    e.params_json,
                    e.prompt_tokens,
                    e.completion_tokens,
                    now,
                    now,
                ),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def touch(self, row_id: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE cache_entries SET last_used_at=?, hit_count=hit_count+1 WHERE id=?",
                (time.time(), row_id),
            )
            self._db.commit()

    def alive_embeddings(self, route: str | None = None):
        # Returns (ids, matrix) for live rows that carry an embedding. Used to
        # build and rebuild the proxima index.
        sql = "SELECT id, embedding FROM cache_entries WHERE alive=1 AND embedding IS NOT NULL"
        args: tuple = ()
        if route is not None:
            sql += " AND route=?"
            args = (route,)
        sql += " ORDER BY id"
        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for row in self._db.execute(sql, args):
            ids.append(row["id"])
            vecs.append(unpack(row["embedding"]))
        if not ids:
            return [], None
        return ids, np.vstack(vecs).astype(np.float32)

    def mark_dead(self, ids: list[int]) -> None:
        with self._lock:
            self._db.executemany(
                "UPDATE cache_entries SET alive=0 WHERE id=?", [(i,) for i in ids]
            )
            self._db.commit()

    def lru_candidates(self, n: int) -> list[int]:
        cur = self._db.execute(
            "SELECT id FROM cache_entries WHERE alive=1 ORDER BY last_used_at ASC LIMIT ?",
            (n,),
        )
        return [r["id"] for r in cur]

    def counts(self) -> dict:
        total = self._db.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        alive = self._db.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE alive=1"
        ).fetchone()[0]
        return {"total": total, "alive": alive, "dead": total - alive}

    def close(self) -> None:
        self._db.close()
