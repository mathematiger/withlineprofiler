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

from lineprofiler.accounting.hardware import describe as describe_hardware
from lineprofiler.accounting.identity import describe as describe_placement
from lineprofiler.accounting.phasetree import PhaseTree, merge_trees, tree_from_dict, tree_to_dict
from lineprofiler.accounting.provenance import describe_source
from lineprofiler.accounting.sampler import (
    Sample,
    open_process,
    read_io_snapshot,
    read_samples,
)
from lineprofiler.accounting.selfio import record_bytes_written
from lineprofiler.accounting.trace import ClockAnchor, Link, Origin, Span, WorkerTrace

FORMAT_VERSION = 1

WORKERS_DIRNAME = "workers"

UNKNOWN_RUN = "unknown"
"""Run id given to worker files written before runs were identified."""


def new_run_id() -> str:
    """Return a sortable, human-readable identifier for one attempt at a run.

    Attempts must be distinguishable because run directories get reused: a requeued Slurm
    job, or a rerun into the same output directory, previously merged the abandoned attempt's
    workers with the new ones and inflated every total with nothing in the report to say so.
    The timestamp prefix makes the newest attempt sort last; the suffix keeps two attempts
    starting in the same second apart.
    """
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


class SnapshotWriter:
    """Writes one worker's aggregate state to ``<run_dir>/workers/w_<pid>_<uuid8>.json``.

    Test specifically:
        - two snapshots in quick succession both leave a complete, parseable file
        - the parent's file is never overwritten by a child process
        - killing the process between snapshots leaves the previous snapshot intact
    """

    def __init__(
        self,
        run_dir: Path,
        role: str = "main",
        run_id: str | None = None,
        source: dict[str, object] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.role = role
        self.run_id = run_id or new_run_id()
        self.placement = describe_placement()
        # Per worker rather than once per run: metadata.json is written by whichever rank wins
        # the race, so on a heterogeneous multi-node job it would describe one node and
        # silently attribute its capacity to every other. Cached module-side, so the repeated
        # constant costs one dict copy per writer rather than a re-enumeration.
        self.hardware = describe_hardware()
        # Sharded by host. One flat directory holding two files per rank is a single-MDT hot
        # spot on Lustre and a slow readdir on NFS, and every rank renames into it on every
        # flush. One subdirectory per node spreads that, and makes "which files came from the
        # node that died" answerable with ls.
        self.worker_dir = run_dir / WORKERS_DIRNAME / str(self.placement.get("host", "unknown"))
        self.pid = os.getpid()
        stem = f"w_{self.run_id}_{self.pid}_{uuid.uuid4().hex[:8]}"
        self.path = self.worker_dir / f"{stem}.json"
        self.samples_path = self.worker_dir / f"{stem}.samples"
        self.trace_path = self.worker_dir / f"{stem}.trace"
        self.started_at = time.time()
        self.write_failures = 0
        self.last_error: str | None = None
        self._backend: dict[str, Any] | None = None
        self._process = open_process()
        try:
            self.worker_dir.mkdir(parents=True, exist_ok=True)
            _write_metadata_once(run_dir, self.run_id, source)
        except OSError as error:
            # An observability tool must never be the reason a twelve-hour job dies. A run
            # directory that cannot be created means no output, not a failed training run.
            self._note_failure(error)

    def write(self, tree: PhaseTree) -> bool:
        """Atomically replace this worker's file with the current aggregate state.

        Returns whether the write landed. A failure is counted and reported in the next
        successful snapshot rather than raised: on shared scratch ``ENOSPC`` and ``EDQUOT``
        are ordinary, and an exception here used to kill the flush thread for the rest of the
        run, freezing the file at a value that still parsed as a complete result.
        """
        payload = {
            "version": FORMAT_VERSION,
            "run_id": self.run_id,
            "pid": self.pid,
            "role": self.role,
            "started_at": self.started_at,
            "written_at": time.time(),
            "backend": self._backend,
            "write_failures": self.write_failures,
            "placement": self.placement,
            "hardware": self.hardware,
            "phases": tree_to_dict(tree),
        }
        document = json.dumps(payload)
        temporary = self.path.with_suffix(".tmp")
        before = read_io_snapshot(self._process)
        try:
            _write_atomic(temporary, self.path, document)
        except OSError as error:
            self._note_failure(error)
            return False
        after = read_io_snapshot(self._process)
        if before.available and after.available:
            record_bytes_written(
                len(document.encode("utf-8")),
                max(0, after.write_bytes - before.write_bytes),
            )
        return True

    def _note_failure(self, error: OSError) -> None:
        self.write_failures += 1
        self.last_error = f"{type(error).__name__}: {error}"

    def record_backend(self, description: dict[str, Any]) -> None:
        """Attach the heavy-profiler artifact description to the next snapshot."""
        self._backend = description

    def append_trace(
        self,
        spans: list[Span],
        links: list[Link],
        paths: list[tuple[str, ...]],
        anchors: list[ClockAnchor],
        dropped: int,
        dropped_links: int,
        origins: dict[int, Origin] | None = None,
    ) -> bool:
        """Append a batch of spans to this worker's trace sidecar.

        Appended rather than atomically replaced, unlike the aggregate snapshot. The two
        differ because the data does: the snapshot is *complete state*, so rewriting it whole
        is what makes a torn write harmless, whereas a trace only grows — rewriting a
        200k-span array on every 30-second flush would turn a cheap snapshot into tens of
        megabytes of churn. A torn final line here costs the last batch and leaves every
        earlier one readable, which :func:`read_trace` relies on.
        """
        if not spans and not links:
            return True
        payload = {
            "paths": ["/".join(path) for path in paths],
            # Keyed by phase id as a string, because JSON object keys are strings and a
            # sparse mapping is what this is — most runs name their phases and have none.
            "origins": {
                str(phase_id): origin.to_dict()
                for phase_id, origin in (origins or {}).items()
            },
            "anchors": [anchor.to_dict() for anchor in anchors],
            "dropped": dropped,
            "dropped_links": dropped_links,
            "spans": [
                [s.phase_id, s.thread_id, s.t0_ns, s.t1_ns, s.cpu_ns, s.flags] for s in spans
            ],
            "links": [link.to_dict() for link in links],
        }
        line = json.dumps(payload) + "\n"
        before = read_io_snapshot(self._process)
        try:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as error:
            self._note_failure(error)
            return False
        after = read_io_snapshot(self._process)
        if before.available and after.available:
            record_bytes_written(
                len(line.encode("utf-8")),
                max(0, after.write_bytes - before.write_bytes),
            )
        return True


def read_trace(path: Path) -> WorkerTrace:
    """Read one worker's trace sidecar, discarding only what is unreadable.

    Every batch is an independent line, so a run killed mid-write loses its last line and
    nothing else. A malformed line is skipped rather than failing the read: the same rule the
    worker files follow — one bad file, or here one bad line, must not cost the whole run's
    trace.

    The interning table and anchors are cumulative across batches; later lines carry the full
    table, so the last one seen wins.
    """
    trace = WorkerTrace()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return trace

    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            batch = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # a torn final line, or a partially flushed one
        try:
            _absorb_batch(trace, batch)
        except (KeyError, TypeError, ValueError):
            continue
    return trace


def _absorb_batch(trace: WorkerTrace, batch: dict[str, Any]) -> None:
    """Fold one sidecar line into the accumulating trace."""
    paths = batch.get("paths")
    if isinstance(paths, list) and len(paths) >= len(trace.paths):
        trace.paths = [tuple(key.split("/")) if key else () for key in paths]
    origins = batch.get("origins")
    if isinstance(origins, dict) and len(origins) >= len(trace.origins):
        trace.origins = {
            int(phase_id): Origin.from_dict(item) for phase_id, item in origins.items()
        }
    anchors = batch.get("anchors")
    if isinstance(anchors, list) and anchors:
        trace.anchors = [ClockAnchor.from_dict(item) for item in anchors]
    trace.dropped = max(trace.dropped, int(batch.get("dropped", 0)))
    trace.dropped_links = max(trace.dropped_links, int(batch.get("dropped_links", 0)))
    for row in batch.get("spans", []):
        trace.spans.append(
            Span(
                phase_id=int(row[0]),
                thread_id=int(row[1]),
                t0_ns=int(row[2]),
                t1_ns=int(row[3]),
                cpu_ns=int(row[4]),
                flags=int(row[5]),
            ),
        )
    for item in batch.get("links", []):
        trace.links.append(Link.from_dict(item))


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
    run_id: str = UNKNOWN_RUN
    placement: dict[str, Any] = dataclass_field(default_factory=dict)
    hardware: dict[str, Any] = dataclass_field(default_factory=dict)
    """This worker's machine capacity. Empty for a file written before it was recorded, which
    is why the report omits the capacity column rather than showing a machine with no cores."""
    write_failures: int = 0
    trace: WorkerTrace = dataclass_field(default_factory=WorkerTrace)
    """Recorded spans and links, empty unless ``merge_run(with_trace=True)`` read them."""

    @property
    def wall_ns(self) -> int:
        """Total wall time this worker recorded at the root of its phase tree."""
        return sum(stats.wall_ns for path, stats in self.tree.items() if len(path) == 1)

    @property
    def host(self) -> str:
        """The node this worker ran on, or ``"?"`` for a file written before hosts existed."""
        return str(self.placement.get("host", "?"))

    @property
    def rank(self) -> int | None:
        """Global rank, when a launcher assigned one."""
        value = self.placement.get("rank")
        return value if isinstance(value, int) else None

    @property
    def label(self) -> str:
        """How to name this worker in a report: rank if it has one, else host and pid."""
        if self.rank is not None:
            return f"rank {self.rank} ({self.host})"
        return f"{self.host}:{self.pid}"


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
    superseded: list[WorkerSnapshot] = dataclass_field(default_factory=list)
    """Workers belonging to an earlier attempt in this same directory. Excluded from every
    total, and named in the report — merging them used to inflate a requeued job silently."""

    @property
    def hosts(self) -> list[str]:
        """Distinct nodes this run touched, in first-seen order.

        Workers written before placement was recorded contribute nothing rather than a ``?``
        entry, so a report over older files falls back to the run metadata's single host
        instead of claiming a node called "?".
        """
        seen: dict[str, None] = {}
        for worker in self.workers:
            if worker.placement.get("host"):
                seen.setdefault(worker.host, None)
        return list(seen)

    @property
    def hardware_by_host(self) -> dict[str, dict[str, Any]]:
        """Each node's capacity, keyed by host, in first-seen order.

        Keyed rather than merged into one figure: a run spanning a 128-core node and a 16-core
        one has no single "machine", and averaging them would describe neither. Hosts whose
        workers recorded nothing are absent, so a partially-upgraded run reports the capacity
        it knows rather than zeros for the rest.
        """
        found: dict[str, dict[str, Any]] = {}
        for worker in self.workers:
            host = worker.host
            if worker.hardware and host not in found:
                found[host] = worker.hardware
        return found

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


def merge_run(
    run_dir: str | Path,
    with_samples: bool = True,
    with_trace: bool = False,
) -> MergedRun:
    """Read every worker file under ``run_dir`` and merge them into one phase tree.

    Args:
        run_dir: The directory passed to ``Profiler(run_dir=...)``.
        with_samples: Read each worker's resource samples too. Set ``False`` to get the phase
            tree alone, which is what makes a large run reportable: samples dominate memory
            by orders of magnitude. A worker samples at 1 Hz, so twelve hours is 43,200 rows
            of ~640 bytes each — about 28 MB per worker held live, or ~1.8 GB across 64 of
            them, and the derived intervals roughly double the peak. The phase trees for the
            same run are a few megabytes in total. Dropping samples costs the I/O, memory and
            GPU blocks and nothing else.
        with_trace: Read each worker's trace sidecar too. Off by default for the same reason
            samples can be dropped — spans outnumber phases by orders of magnitude — so the
            ordinary report pays nothing for files it does not draw.

    Test specifically:
        - the merge is unaffected by the order in which worker files are read
        - a truncated or empty worker file is reported in ``unreadable``, not raised
        - summed counters equal the known total work across a spawn/fork/forkserver matrix
        - ``with_samples=False`` yields the same phase tree and no samples
    """
    run_path = Path(run_dir)
    found: list[WorkerSnapshot] = []
    unreadable: list[Path] = []

    for path in sorted((run_path / WORKERS_DIRNAME).rglob("w_*.json")):
        worker = _read_worker(path, with_samples=with_samples)
        if worker is None:
            unreadable.append(path)
            continue
        if with_trace:
            worker.trace = read_trace(path.with_suffix(".trace"))
        found.append(worker)

    workers, superseded = _split_by_attempt(found)
    merged: PhaseTree = {}
    for worker in workers:
        merge_trees(merged, worker.tree)

    return MergedRun(
        tree=merged,
        workers=workers,
        unreadable=unreadable,
        metadata=_read_metadata(run_path),
        superseded=superseded,
    )


def _split_by_attempt(
    workers: list[WorkerSnapshot],
) -> tuple[list[WorkerSnapshot], list[WorkerSnapshot]]:
    """Return (newest attempt, everything older) from workers sharing a run directory.

    The newest attempt is the run id whose first worker started last. Files predating run ids
    carry ``UNKNOWN_RUN`` and are treated as one attempt among the others, so an old run
    directory still reports rather than vanishing.
    """
    by_run: dict[str, list[WorkerSnapshot]] = {}
    for worker in workers:
        by_run.setdefault(worker.run_id, []).append(worker)
    if len(by_run) <= 1:
        return workers, []

    newest = max(by_run, key=lambda run_id: max(w.started_at for w in by_run[run_id]))
    older = [w for run_id, group in by_run.items() if run_id != newest for w in group]
    return by_run[newest], older


def _read_worker(path: Path, with_samples: bool = True) -> WorkerSnapshot | None:
    """Parse one worker file and its samples, returning ``None`` if the file is unusable.

    Everything after the JSON parse is guarded too, not just the parse. A file that is valid
    JSON but structurally wrong — a version skew, a foreign file matching the glob, a half
    page that happens to close its braces — used to raise straight out of ``merge_run`` and
    abort the whole report. One bad file out of sixty-four is a lost worker, not a lost run.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return WorkerSnapshot(
            pid=int(payload["pid"]),
            role=str(payload.get("role", "main")),
            path=path,
            started_at=float(payload["started_at"]),
            written_at=float(payload["written_at"]),
            tree=tree_from_dict(payload["phases"]),
            samples=read_samples(path.with_suffix(".samples")) if with_samples else [],
            backend=payload.get("backend"),
            run_id=str(payload.get("run_id", UNKNOWN_RUN)),
            placement=payload.get("placement") or {},
            hardware=payload.get("hardware") or {},
            write_failures=int(payload.get("write_failures", 0)),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _write_atomic(temporary: Path, target: Path, document: str) -> None:
    """Write to ``temporary``, force it to stable storage, then rename it over ``target``.

    ``os.replace`` is atomic on ext4, XFS, NFS and Lustre alike, which is what makes a torn
    write survivable: the previous complete file stays in place. Atomicity is not durability,
    though. Without the ``fsync`` the rename can reach the disk while the data behind it has
    not, so a node that loses power leaves a zero-length or garbage file — and node death,
    not process death, is the failure that actually happens on a cluster. The parent
    directory is synced as well, or the rename itself is not guaranteed to survive.
    """
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(handle, document.encode("utf-8"))
        os.fsync(handle)
    finally:
        os.close(handle)
    os.replace(temporary, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_metadata_once(
    run_dir: Path,
    run_id: str,
    source: dict[str, object] | None = None,
) -> None:
    """Record run-level context, once per attempt.

    Ranks of the same attempt skip the write after the first; a *new* attempt into the same
    directory overwrites, so the header describes the run you just did rather than one from
    last week. The write goes through a uniquely-named temporary and ``os.replace``, so ranks
    starting simultaneously cannot interleave into one another's output. ``exists()`` alone
    was not enough: on a shared filesystem its answer is cached, so on a thousand-rank launch
    many ranks pass the check and then write the same path at once, leaving JSON that parses
    as nothing and a report that prints ``Host ?``.

    ``source`` overrides the git lookup, for an embedding program that knows its own revision
    — or wants to record a config hash instead, a config change altering behaviour as much as
    a code change does.
    """
    path = run_dir / "metadata.json"
    if _read_metadata(path.parent).get("run_id") == run_id:
        return
    # After the dedupe, so only the rank that actually writes pays for the subprocess.
    if source is None:
        source = describe_source()
    payload = {
        "version": FORMAT_VERSION,
        "run_id": run_id,
        "started_at": time.time(),
        # Both clocks at the same instant, so a Chrome trace written by a heavy backend on
        # the monotonic clock can be lined up with our epoch-stamped resource samples.
        "epoch_anchor": {"time": time.time(), "perf_counter_ns": time.perf_counter_ns()},
        "argv": sys.argv,
        "host": socket.gethostname(),
        "python": sys.version,
        "source": source,
    }
    document = json.dumps(payload, indent=2)
    process = open_process()
    temporary = run_dir / f".metadata.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with contextlib.suppress(OSError):
        before = read_io_snapshot(process)
        _write_atomic(temporary, path, document)
        after = read_io_snapshot(process)
        if before.available and after.available:
            record_bytes_written(
                len(document.encode("utf-8")),
                max(0, after.write_bytes - before.write_bytes),
            )
    with contextlib.suppress(OSError):
        temporary.unlink(missing_ok=True)


def _read_metadata(run_dir: Path) -> dict[str, Any]:
    try:
        loaded: dict[str, Any] = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded
