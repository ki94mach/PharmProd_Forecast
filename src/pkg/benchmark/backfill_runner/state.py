"""SQLite job state, identity, and run lock for unattended backfills."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_SUCCESS = "SUCCESS"
JOB_FAILED = "FAILED"

JOB_STATUSES = frozenset({JOB_PENDING, JOB_RUNNING, JOB_SUCCESS, JOB_FAILED})

LOCK_FILENAME = "run.lock"
DB_FILENAME = "backfill.sqlite"


class BackfillStateError(Exception):
    """Checkpoint / lock / identity error."""


class RunLockError(BackfillStateError):
    """Another process already holds the experiment run lock."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_git_commit() -> str:
    """Best-effort short git SHA for the repository; ``unknown`` if unavailable."""
    try:
        root = Path(__file__).resolve().parents[4]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()[:40] or "unknown"
    except Exception:
        return "unknown"


def compute_config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of experiment configuration (sorted JSON)."""
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_experiment_id(vintage_name: str, universe_name: str, engine: str) -> str:
    """Default experiment id: vintage__universe__engine."""
    return f"{vintage_name}__{universe_name}__{engine}"


@dataclass(frozen=True)
class JobIdentity:
    """Canonical identity for one SKU × vintage job."""

    experiment_id: str
    engine_version: str
    config_hash: str
    quarter: str
    forecast_origin: int
    product_id: str

    @property
    def job_id(self) -> str:
        raw = "|".join(
            [
                self.experiment_id,
                self.engine_version,
                self.config_hash,
                self.quarter,
                str(int(self.forecast_origin)),
                self.product_id,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    @property
    def slug(self) -> str:
        safe = self.product_id.replace("/", "_").replace("\\", "_")
        return f"{self.quarter}__{safe}"


@dataclass
class JobRecord:
    identity: JobIdentity
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_seconds: Optional[float] = None
    attempt_count: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    selected_model: Optional[str] = None
    git_commit: Optional[str] = None
    updated_at: Optional[str] = None


class RunLock:
    """Exclusive lock for one experiment directory (no accidental dual runners)."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.lock_path = self.experiment_dir / LOCK_FILENAME
        self._held = False

    def acquire(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        if self._held:
            raise RunLockError(f"Run lock already held by this process: {self.lock_path}")
        if self.lock_path.exists():
            meta = self._read_lock()
            pid = int(meta.get("pid", -1))
            # Any live holder blocks — including same PID (second runner in-process).
            if pid > 0 and _pid_is_alive(pid):
                raise RunLockError(
                    f"Experiment locked by pid={pid} "
                    f"started_at={meta.get('started_at')} path={self.lock_path}"
                )
            # Stale lock from a dead process — reclaim.
        payload = {
            "pid": os.getpid(),
            "started_at": _utc_stamp(),
            "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "",
        }
        tmp = self.lock_path.with_suffix(".lock.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.lock_path)
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        try:
            meta = self._read_lock()
            if int(meta.get("pid", -1)) == os.getpid() and self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass
        self._held = False

    def _read_lock(self) -> dict[str, Any]:
        try:
            return json.loads(self.lock_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            # Windows: OpenProcess via ctypes would be ideal; os.kill(pid, 0)
            # raises OSError if the process does not exist on modern CPython.
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


class JobStateStore:
    """Transactional SQLite checkpoint store for SKU-vintage jobs."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.experiment_dir / DB_FILENAME
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    vintage_manifest TEXT NOT NULL,
                    universe_manifest TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    git_commit TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    quarter TEXT NOT NULL,
                    forecast_origin INTEGER NOT NULL,
                    product_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    runtime_seconds REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT,
                    error_message TEXT,
                    output_path TEXT,
                    selected_model TEXT,
                    git_commit TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE (
                        experiment_id,
                        engine_version,
                        config_hash,
                        quarter,
                        forecast_origin,
                        product_id
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON jobs (experiment_id, status);
                CREATE INDEX IF NOT EXISTS idx_jobs_quarter_product
                    ON jobs (experiment_id, quarter, product_id);
                """
            )
            conn.commit()

    def upsert_experiment(
        self,
        *,
        experiment_id: str,
        vintage_manifest: str,
        universe_manifest: str,
        engine_version: str,
        config: dict[str, Any],
        config_hash: str,
        git_commit: str,
    ) -> None:
        now = _utc_stamp()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, vintage_manifest, universe_manifest,
                    engine_version, config_json, config_hash, git_commit,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    config_json=excluded.config_json,
                    config_hash=excluded.config_hash,
                    git_commit=excluded.git_commit
                """,
                (
                    experiment_id,
                    vintage_manifest,
                    universe_manifest,
                    engine_version,
                    json.dumps(config, sort_keys=True, default=str),
                    config_hash,
                    git_commit,
                    now,
                    now,
                ),
            )
            conn.commit()

    def ensure_job(self, identity: JobIdentity, *, git_commit: str) -> JobRecord:
        """Insert PENDING job if missing; return current record."""
        now = _utc_stamp()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, experiment_id, engine_version, config_hash,
                        quarter, forecast_origin, product_id, status,
                        attempt_count, git_commit, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        identity.job_id,
                        identity.experiment_id,
                        identity.engine_version,
                        identity.config_hash,
                        identity.quarter,
                        int(identity.forecast_origin),
                        identity.product_id,
                        JOB_PENDING,
                        git_commit,
                        now,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
                ).fetchone()
            return self._row_to_record(row)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def list_jobs(
        self,
        experiment_id: str,
        *,
        statuses: Optional[Sequence[str]] = None,
        quarter: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> list[JobRecord]:
        clauses = ["experiment_id = ?"]
        params: list[Any] = [experiment_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if quarter:
            clauses.append("quarter = ?")
            params.append(quarter)
        if product_id:
            clauses.append("product_id = ?")
            params.append(product_id)
        sql = "SELECT * FROM jobs WHERE " + " AND ".join(clauses) + " ORDER BY quarter, product_id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def status_counts(self, experiment_id: str) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM jobs
                WHERE experiment_id = ?
                GROUP BY status
                """,
                (experiment_id,),
            ).fetchall()
        counts = {s: 0 for s in JOB_STATUSES}
        for row in rows:
            counts[str(row["status"])] = int(row["n"])
        return counts

    def reclaim_stale_running(
        self,
        experiment_id: str,
        *,
        quarter: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> int:
        """Mark RUNNING jobs as PENDING so resume can retry after a crash."""
        now = _utc_stamp()
        clauses = ["experiment_id = ?", "status = ?"]
        params: list[Any] = [experiment_id, JOB_RUNNING]
        if quarter:
            clauses.append("quarter = ?")
            params.append(quarter)
        if product_id:
            clauses.append("product_id = ?")
            params.append(product_id)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE jobs
                SET status = ?,
                    error_type = 'StaleRunning',
                    error_message = 'Reclaimed stale RUNNING after process interruption',
                    updated_at = ?
                WHERE {where}
                """,
                [JOB_PENDING, now, *params],
            )
            conn.commit()
            return int(cur.rowcount)

    def mark_running(self, identity: JobIdentity, *, git_commit: str) -> JobRecord:
        now = _utc_stamp()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            if row is None:
                raise BackfillStateError(f"unknown job_id={identity.job_id}")
            if row["status"] == JOB_SUCCESS:
                raise BackfillStateError(
                    f"SUCCESS job is immutable without --force-job: {identity.slug}"
                )
            attempt = int(row["attempt_count"]) + 1
            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    started_at = ?,
                    finished_at = NULL,
                    runtime_seconds = NULL,
                    attempt_count = ?,
                    error_type = NULL,
                    error_message = NULL,
                    selected_model = NULL,
                    git_commit = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_RUNNING, now, attempt, git_commit, now, identity.job_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            return self._row_to_record(row)

    def try_claim_job(
        self, identity: JobIdentity, *, git_commit: str
    ) -> Optional[JobRecord]:
        """Atomically claim a runnable job (``PENDING``/``FAILED`` → ``RUNNING``).

        Returns the updated record on success, or ``None`` if another worker
        already claimed it (or the job is SUCCESS/RUNNING). Uses
        ``BEGIN IMMEDIATE`` so concurrent workers cannot claim the same job.
        """
        now = _utc_stamp()
        claimable = (JOB_PENDING, JOB_FAILED)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            if row is None or str(row["status"]) not in claimable:
                conn.execute("ROLLBACK")
                return None
            prior = str(row["status"])
            attempt = int(row["attempt_count"] or 0) + 1
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    started_at = ?,
                    finished_at = NULL,
                    runtime_seconds = NULL,
                    attempt_count = ?,
                    error_type = NULL,
                    error_message = NULL,
                    selected_model = NULL,
                    git_commit = ?,
                    updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    JOB_RUNNING,
                    now,
                    attempt,
                    git_commit,
                    now,
                    identity.job_id,
                    prior,
                ),
            )
            if int(cur.rowcount) != 1:
                conn.execute("ROLLBACK")
                return None
            conn.commit()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            return self._row_to_record(row)

    def mark_success(
        self,
        identity: JobIdentity,
        *,
        output_path: str,
        selected_model: Optional[str],
        started_at: str,
        finished_at: str,
        runtime_seconds: float,
        git_commit: str,
    ) -> JobRecord:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    finished_at = ?,
                    runtime_seconds = ?,
                    error_type = NULL,
                    error_message = NULL,
                    output_path = ?,
                    selected_model = ?,
                    started_at = COALESCE(started_at, ?),
                    git_commit = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JOB_SUCCESS,
                    finished_at,
                    float(runtime_seconds),
                    output_path,
                    selected_model,
                    started_at,
                    git_commit,
                    finished_at,
                    identity.job_id,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            return self._row_to_record(row)

    def mark_failed(
        self,
        identity: JobIdentity,
        *,
        error_type: Optional[str],
        error_message: Optional[str],
        started_at: str,
        finished_at: str,
        runtime_seconds: float,
        git_commit: str,
        output_path: Optional[str] = None,
    ) -> JobRecord:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    finished_at = ?,
                    runtime_seconds = ?,
                    error_type = ?,
                    error_message = ?,
                    output_path = ?,
                    started_at = COALESCE(started_at, ?),
                    git_commit = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JOB_FAILED,
                    finished_at,
                    float(runtime_seconds),
                    error_type,
                    error_message,
                    output_path,
                    started_at,
                    git_commit,
                    finished_at,
                    identity.job_id,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            return self._row_to_record(row)

    def reset_for_force(
        self,
        identity: JobIdentity,
        *,
        git_commit: str,
    ) -> JobRecord:
        """Force SUCCESS/FAILED/RUNNING back to PENDING for recompute."""
        now = _utc_stamp()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    started_at = NULL,
                    finished_at = NULL,
                    runtime_seconds = NULL,
                    error_type = NULL,
                    error_message = NULL,
                    output_path = NULL,
                    selected_model = NULL,
                    git_commit = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_PENDING, git_commit, now, identity.job_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (identity.job_id,)
            ).fetchone()
            return self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        identity = JobIdentity(
            experiment_id=str(row["experiment_id"]),
            engine_version=str(row["engine_version"]),
            config_hash=str(row["config_hash"]),
            quarter=str(row["quarter"]),
            forecast_origin=int(row["forecast_origin"]),
            product_id=str(row["product_id"]),
        )
        return JobRecord(
            identity=identity,
            status=str(row["status"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            runtime_seconds=row["runtime_seconds"],
            attempt_count=int(row["attempt_count"] or 0),
            error_type=row["error_type"],
            error_message=row["error_message"],
            output_path=row["output_path"],
            selected_model=row["selected_model"],
            git_commit=row["git_commit"],
            updated_at=row["updated_at"],
        )


def should_run_job(
    record: JobRecord,
    *,
    resume: bool,
    retry_failed: bool,
    force_job: bool,
) -> bool:
    """Decide whether a job should execute under the current CLI flags."""
    if force_job:
        return True
    if record.status == JOB_SUCCESS:
        return False
    if record.status == JOB_FAILED:
        return bool(retry_failed or not resume)
    if record.status == JOB_RUNNING:
        # Caller should reclaim stale RUNNING first; if still RUNNING, skip.
        return False
    # PENDING
    return True
