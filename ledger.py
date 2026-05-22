"""Submission ledger backed by SQLite.

The ledger is the source of truth for everything we ship to Codabench:

- one row per built submission ZIP (uniquely identified by ``manifest_sha``,
  the sha256 of the deterministic ZIP contents),
- offline validation numbers from the round simulator,
- leaderboard scores once a round resolves,
- the git context the ZIP was built from, so the best-scoring run's
  source files at the repo root can be reproduced bit-for-bit.

This module exposes a tiny API used by ``submit.py`` and the validation
harness; the schema lives in :func:`init_db`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LEDGER_PATH = Path(__file__).resolve().parent / "runs" / "ledger.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   TEXT    NOT NULL,
    branch               TEXT,
    commit_sha           TEXT,
    manifest_sha         TEXT    NOT NULL,
    zip_path             TEXT    NOT NULL,
    model_name           TEXT    NOT NULL,
    hyperparams_json     TEXT,
    val_nll_mean         REAL,
    val_nll_std          REAL,
    val_auc              REAL,
    leaderboard_round_id TEXT,
    leaderboard_nll      REAL,
    leaderboard_auc      REAL,
    uploaded_at          TEXT,
    notes                TEXT
);

CREATE INDEX IF NOT EXISTS idx_subs_manifest ON submissions(manifest_sha);
CREATE INDEX IF NOT EXISTS idx_subs_model    ON submissions(model_name);
CREATE INDEX IF NOT EXISTS idx_subs_ts       ON submissions(ts);
"""


@dataclass
class SubmissionRow:
    id: int
    ts: str
    branch: str | None
    commit_sha: str | None
    manifest_sha: str
    zip_path: str
    model_name: str
    hyperparams_json: str | None
    val_nll_mean: float | None
    val_nll_std: float | None
    val_auc: float | None
    leaderboard_round_id: str | None
    leaderboard_nll: float | None
    leaderboard_auc: float | None
    uploaded_at: str | None
    notes: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SubmissionRow":
        return cls(**{k: row[k] for k in row.keys()})

    @property
    def hyperparams(self) -> dict[str, Any]:
        return json.loads(self.hyperparams_json) if self.hyperparams_json else {}


def init_db(path: Path = LEDGER_PATH) -> Path:
    """Create the ledger DB and schema if missing. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


@contextmanager
def connect(path: Path = LEDGER_PATH):
    """Context manager that yields a connection with row factory set."""
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_submission(
    *,
    ts: str,
    manifest_sha: str,
    zip_path: str,
    model_name: str,
    branch: str | None = None,
    commit_sha: str | None = None,
    hyperparams: dict[str, Any] | None = None,
    val_nll_mean: float | None = None,
    val_nll_std: float | None = None,
    val_auc: float | None = None,
    notes: str | None = None,
    path: Path = LEDGER_PATH,
) -> int:
    """Insert a new submission row. Returns the row id."""
    with connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO submissions (
                ts, branch, commit_sha, manifest_sha, zip_path,
                model_name, hyperparams_json,
                val_nll_mean, val_nll_std, val_auc, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                branch,
                commit_sha,
                manifest_sha,
                zip_path,
                model_name,
                json.dumps(hyperparams) if hyperparams is not None else None,
                val_nll_mean,
                val_nll_std,
                val_auc,
                notes,
            ),
        )
        return int(cur.lastrowid)


def update_score(
    submission_id: int,
    *,
    leaderboard_round_id: str | None = None,
    leaderboard_nll: float | None = None,
    leaderboard_auc: float | None = None,
    uploaded_at: str | None = None,
    notes_append: str | None = None,
    path: Path = LEDGER_PATH,
) -> None:
    """Patch leaderboard fields for an existing submission row.

    Only non-None values are written. If ``notes_append`` is given, the new
    text is appended (with a newline) to whatever notes already exist.
    """
    sets: list[str] = []
    params: list[Any] = []
    for col, val in (
        ("leaderboard_round_id", leaderboard_round_id),
        ("leaderboard_nll", leaderboard_nll),
        ("leaderboard_auc", leaderboard_auc),
        ("uploaded_at", uploaded_at),
    ):
        if val is not None:
            sets.append(f"{col} = ?")
            params.append(val)
    with connect(path) as conn:
        if notes_append is not None:
            cur = conn.execute(
                "SELECT notes FROM submissions WHERE id = ?", (submission_id,)
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"No submission with id={submission_id}")
            existing = row["notes"] or ""
            merged = (existing + "\n" + notes_append).strip()
            sets.append("notes = ?")
            params.append(merged)
        if not sets:
            return
        params.append(submission_id)
        conn.execute(
            f"UPDATE submissions SET {', '.join(sets)} WHERE id = ?", params
        )


def list_submissions(
    *,
    model_name: str | None = None,
    only_scored: bool = False,
    limit: int | None = None,
    path: Path = LEDGER_PATH,
) -> list[SubmissionRow]:
    """Return submissions ordered by ts DESC, optionally filtered."""
    sql = "SELECT * FROM submissions"
    where: list[str] = []
    params: list[Any] = []
    if model_name is not None:
        where.append("model_name = ?")
        params.append(model_name)
    if only_scored:
        where.append("leaderboard_nll IS NOT NULL")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [SubmissionRow.from_row(r) for r in rows]


def best_submission(path: Path = LEDGER_PATH) -> SubmissionRow | None:
    """Return the submission with the highest leaderboard NLL (higher is better)."""
    with connect(path) as conn:
        row = conn.execute(
            """
            SELECT * FROM submissions
            WHERE leaderboard_nll IS NOT NULL
            ORDER BY leaderboard_nll DESC
            LIMIT 1
            """
        ).fetchone()
        return SubmissionRow.from_row(row) if row else None


def find_by_manifest(
    manifest_sha: str, path: Path = LEDGER_PATH
) -> list[SubmissionRow]:
    """Return all rows that share a manifest hash (i.e., identical ZIP contents)."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM submissions WHERE manifest_sha = ? ORDER BY ts DESC",
            (manifest_sha,),
        ).fetchall()
        return [SubmissionRow.from_row(r) for r in rows]


__all__: Iterable[str] = (
    "LEDGER_PATH",
    "SubmissionRow",
    "best_submission",
    "connect",
    "find_by_manifest",
    "init_db",
    "list_submissions",
    "record_submission",
    "update_score",
)
