"""Crash-resilient persistence of aggregate state, and the offline merge across workers.

Each process writes its *complete* aggregate state to one file, replacing it atomically on
every snapshot. A run that is killed at hour eleven of twelve therefore still yields a file
holding everything up to the last flush — there is no log to replay and no partial line to
discard, because a torn write leaves the previous complete file untouched.

Worker files are named ``w_<pid>_<uuid8>.json``. The uuid matters: an actor process that
dies and is respawned reuses its rank but not its pid, and a plain counter would collide
across ``spawn``.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from lineprofiler.accounting.phase import PhaseTree, merge_trees, tree_from_dict, tree_to_dict
from lineprofiler.accounting.sampler import (
    Sample,
    open_process,
    read_io_snapshot,
    read_samples,
)
from lineprofiler.accounting.selfio import record_bytes_written

FORMAT_VERSION = 1

WORKERS_DIRNAME = "workers"


class SnapshotWriter:
    """Writes one worker's aggregate state to ``<run_dir>/workers/w_<pid>_<uuid8>.json``.

    Test specifically:
        - two snapshots in quick succession both leave a complete, parseable file
        - the parent's file is never overwritten by a child process
        - killing the process between snapshots leaves the previous snapshot intact
    """

    def __init__(self, run_dir: Path, role: str = "main") -> None:
        self.run_dir = run_dir
        self.role = role
        self.worker_dir = run_dir / WORKERS_DIRNAME
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        stem = f"w_{self.pid}_{uuid.uuid4().hex[:8]}"
        self.path = self.worker_dir / f"{stem}.json"
        self.samples_path = self.worker_dir / f"{stem}.samples"
        self.started_at = time.time()
        self._backend: dict[str, Any] | None = None
        self._process = open_process()
        _write_metadata_once(run_dir)

    def write(self, tree: PhaseTree) -> None:
        """Atomically replace this worker's file with the current aggregate state."""
        payload = {
            "version": FORMAT_VERSION,
            "pid": self.pid,
            "role": self.role,
            "started_at": self.started_at,
            "written_at": time.time(),
            "backend": self._backend,
            "phases": tree_to_dict(tree),
        }
        document = json.dumps(payload)
        temporary = self.path.with_suffix(".tmp")
        before = read_io_snapshot(self._process)
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, self.path)
        after = read_io_snapshot(self._process)
        record_bytes_written(
            len(document.encode("utf-8")),
            max(0, after.write_bytes - before.write_bytes),
        )

    def record_backend(self, description: dict[str, Any]) -> None:
        """Attach the heavy-profiler artifact description to the next snapshot."""
        self._backend = description


@dataclass(slots=True)
class WorkerSnapshot:
    """One worker's state as read back from disk."""

    pid: int
    role: str
    path: Path
    started_at: float
    written_at: float
    tree: PhaseTree
    samples: list[Sample] = dataclass_field(default_factory=list)
    backend: dict[str, Any] | None = None

    @property
    def wall_ns(self) -> int:
        """Total wall time this worker recorded at the root of its phase tree."""
        return sum(stats.wall_ns for path, stats in self.tree.items() if len(path) == 1)


@dataclass(slots=True)
class MergedRun:
    """Every worker of one run, merged, plus what could not be read.

    ``unreadable`` is reported rather than hidden: a worker killed with ``SIGKILL`` before
    its first snapshot leaves nothing, and a run that silently drops a worker's work would
    misreport both totals and the imbalance ratio.
    """

    tree: PhaseTree
    workers: list[WorkerSnapshot]
    unreadable: list[Path]
    metadata: dict[str, Any]

    @property
    def imbalance(self) -> float:
        """``max(T_i) / mean(T_i)`` over per-worker wall time; 1.0 means perfectly even."""
        return imbalance_of(self.workers)

    @property
    def roles(self) -> list[str]:
        """Distinct roles present, ordered by total wall time descending."""
        totals: dict[str, int] = {}
        for worker in self.workers:
            totals[worker.role] = totals.get(worker.role, 0) + worker.wall_ns
        return sorted(totals, key=lambda role: -totals[role])

    def workers_of(self, role: str) -> list[WorkerSnapshot]:
        return [worker for worker in self.workers if worker.role == role]

    def tree_of(self, role: str) -> PhaseTree:
        """Merge only the workers with the given role.

        Test specifically:
            - a role's tree excludes every other role's phases
            - an unknown role returns an empty tree rather than raising
        """
        merged: PhaseTree = {}
        for worker in self.workers_of(role):
            merge_trees(merged, worker.tree)
        return merged

    def samples_of(self, role: str) -> list[Sample]:
        samples: list[Sample] = []
        for worker in self.workers_of(role):
            samples.extend(worker.samples)
        return samples

    def samples_by_process(self) -> list[list[Sample]]:
        """Each worker's samples, kept separate.

        Cumulative OS counters may only be differenced within one process, so callers must
        never flatten these into a single series.
        """
        return [worker.samples for worker in self.workers if worker.samples]

    def backend_artifacts(self) -> list[dict[str, Any]]:
        """Every heavy-profiler artifact recorded by any worker."""
        return [w.backend for w in self.workers if w.backend and w.backend.get("artifact")]


def imbalance_of(workers: list[WorkerSnapshot]) -> float:
    """``max(T_i) / mean(T_i)`` over wall time; 1.0 is perfectly even, 2.0 is one worker
    doing twice the average.

    Test specifically:
        - an empty list and a single worker both give 1.0
        - two workers at 3:7 give 1.4
    """
    totals = [worker.wall_ns for worker in workers if worker.wall_ns > 0]
    if not totals:
        return 1.0
    return max(totals) / (sum(totals) / len(totals))


def merge_run(run_dir: str | Path) -> MergedRun:
    """Read every worker file under ``run_dir`` and merge them into one phase tree.

    Test specifically:
        - the merge is unaffected by the order in which worker files are read
        - a truncated or empty worker file is reported in ``unreadable``, not raised
        - summed counters equal the known total work across a spawn/fork/forkserver matrix
    """
    run_path = Path(run_dir)
    merged: PhaseTree = {}
    workers: list[WorkerSnapshot] = []
    unreadable: list[Path] = []

    for path in sorted((run_path / WORKERS_DIRNAME).glob("w_*.json")):
        worker = _read_worker(path)
        if worker is None:
            unreadable.append(path)
            continue
        workers.append(worker)
        merge_trees(merged, worker.tree)

    return MergedRun(
        tree=merged,
        workers=workers,
        unreadable=unreadable,
        metadata=_read_metadata(run_path),
    )


def _read_worker(path: Path) -> WorkerSnapshot | None:
    """Parse one worker file and its samples, returning ``None`` if the file is unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return WorkerSnapshot(
        pid=payload["pid"],
        role=payload.get("role", "main"),
        path=path,
        started_at=payload["started_at"],
        written_at=payload["written_at"],
        tree=tree_from_dict(payload["phases"]),
        samples=read_samples(path.with_suffix(".samples")),
        backend=payload.get("backend"),
    )


def _write_metadata_once(run_dir: Path) -> None:
    """Record run-level context. The first process to arrive wins; children do not overwrite."""
    path = run_dir / "metadata.json"
    if path.exists():
        return
    payload = {
        "version": FORMAT_VERSION,
        "started_at": time.time(),
        # Both clocks at the same instant, so a Chrome trace written by a heavy backend on
        # the monotonic clock can be lined up with our epoch-stamped resource samples.
        "epoch_anchor": {"time": time.time(), "perf_counter_ns": time.perf_counter_ns()},
        "argv": sys.argv,
        "host": socket.gethostname(),
        "python": sys.version,
    }
    document = json.dumps(payload, indent=2)
    process = open_process()
    with contextlib.suppress(OSError):
        before = read_io_snapshot(process)
        path.write_text(document, encoding="utf-8")
        after = read_io_snapshot(process)
        record_bytes_written(
            len(document.encode("utf-8")),
            max(0, after.write_bytes - before.write_bytes),
        )


def _read_metadata(run_dir: Path) -> dict[str, Any]:
    try:
        loaded: dict[str, Any] = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded
