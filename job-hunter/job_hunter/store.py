"""SQLite store: every job seen, its match, and every application — so nothing is ever applied to twice."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .jobs import Job
from .matcher import Match

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, source TEXT, title TEXT, company TEXT, location TEXT, work_mode TEXT,
    url TEXT, apply_url TEXT, posted_at TEXT, salary TEXT, description TEXT, emails TEXT,
    dedupe_key TEXT, first_seen TEXT, last_seen TEXT
);
CREATE INDEX IF NOT EXISTS jobs_dedupe ON jobs(dedupe_key);
CREATE TABLE IF NOT EXISTS matches (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id), score REAL, accepted INTEGER, details TEXT, scored_at TEXT
);
CREATE TABLE IF NOT EXISTS applications (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id), status TEXT, channel TEXT, recipient TEXT,
    subject TEXT, body TEXT, created_at TEXT, sent_at TEXT, note TEXT
);
"""
# drafted -> sent | failed ; manual -> applied (after you submit the form) ; any -> skipped
STATUSES = ("drafted", "sent", "failed", "manual", "applied", "skipped", "interview", "rejected", "offer")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def upsert_jobs(self, jobs: list[Job]) -> int:
        """Insert new jobs, refresh known ones. Returns how many were new (cross-posted copies don't count)."""
        new = 0
        ts = now()
        for j in jobs:
            exists = self.db.execute("SELECT 1 FROM jobs WHERE id = ?", (j.id,)).fetchone()
            if exists:
                self.db.execute("UPDATE jobs SET last_seen = ? WHERE id = ?", (ts, j.id))
                continue
            if self.db.execute("SELECT 1 FROM jobs WHERE dedupe_key = ?", (j.dedupe_key,)).fetchone():
                continue  # same opening already stored from another source
            self.db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (j.id, j.source, j.title, j.company, j.location, j.work_mode, j.url, j.apply_url, j.posted_at,
                 j.salary, j.description, json.dumps(j.emails), j.dedupe_key, ts, ts))
            new += 1
        self.db.commit()
        return new

    def jobs(self, unscored_only: bool = False) -> list[Job]:
        q = "SELECT * FROM jobs"
        if unscored_only:
            q += " WHERE id NOT IN (SELECT job_id FROM matches)"
        return [self._job(r) for r in self.db.execute(q)]

    def job(self, job_id: str) -> Job | None:
        r = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job(r) if r else None

    @staticmethod
    def _job(r: sqlite3.Row) -> Job:
        return Job(r["source"], r["id"].split(":", 1)[1], r["title"], r["company"], r["location"], r["work_mode"],
                   r["url"], r["description"], r["apply_url"], r["posted_at"], r["salary"], json.loads(r["emails"]))

    def save_matches(self, matches: list[Match]) -> None:
        ts = now()
        self.db.executemany("INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?)",
                            [(m.job_id, m.score, int(m.accepted), json.dumps(m.to_dict()), ts) for m in matches])
        self.db.commit()

    def shortlist(self, limit: int | None = None, include_applied: bool = False) -> list[tuple[Job, Match]]:
        q = ("SELECT j.*, m.details FROM jobs j JOIN matches m ON m.job_id = j.id WHERE m.accepted = 1"
             + ("" if include_applied else " AND j.id NOT IN (SELECT job_id FROM applications)")
             + " ORDER BY m.score DESC" + (f" LIMIT {int(limit)}" if limit else ""))
        return [(self._job(r), Match(**json.loads(r["details"]))) for r in self.db.execute(q)]

    def record_application(self, job_id: str, status: str, channel: str, recipient: str = "",
                           subject: str = "", body: str = "", note: str = "") -> None:
        assert status in STATUSES, status
        sent = now() if status == "sent" else None
        self.db.execute(
            "INSERT INTO applications VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET "
            "status=excluded.status, channel=excluded.channel, recipient=excluded.recipient, "
            "subject=excluded.subject, body=excluded.body, sent_at=COALESCE(excluded.sent_at, sent_at), "
            "note=excluded.note",
            (job_id, status, channel, recipient, subject, body, now(), sent, note))
        self.db.commit()

    def set_status(self, job_id: str, status: str, note: str = "") -> bool:
        assert status in STATUSES, status
        cur = self.db.execute("UPDATE applications SET status = ?, note = ? WHERE job_id = ?", (status, note, job_id))
        if not cur.rowcount and self.job(job_id):
            self.record_application(job_id, status, "manual", note=note)
            return True
        self.db.commit()
        return bool(cur.rowcount)

    def application_status(self, job_id: str) -> str | None:
        r = self.db.execute("SELECT status FROM applications WHERE job_id = ?", (job_id,)).fetchone()
        return r["status"] if r else None

    def sent_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        return self.db.execute("SELECT COUNT(*) FROM applications WHERE status = 'sent' AND sent_at >= ?",
                               (today,)).fetchone()[0]

    def contacted(self, recipient: str) -> bool:
        return bool(self.db.execute("SELECT 1 FROM applications WHERE status = 'sent' AND recipient = ?",
                                    (recipient,)).fetchone())

    def applications(self) -> list[sqlite3.Row]:
        return list(self.db.execute(
            "SELECT a.*, j.title, j.company, j.work_mode, j.url, m.score FROM applications a "
            "JOIN jobs j ON j.id = a.job_id LEFT JOIN matches m ON m.job_id = a.job_id ORDER BY a.created_at DESC"))

    def counts(self) -> dict[str, int]:
        c = {"jobs": self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
             "matched": self.db.execute("SELECT COUNT(*) FROM matches WHERE accepted = 1").fetchone()[0]}
        for r in self.db.execute("SELECT status, COUNT(*) n FROM applications GROUP BY status"):
            c[r["status"]] = r["n"]
        return c
