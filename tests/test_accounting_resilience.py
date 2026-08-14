"""Regression tests for failures that used to produce wrong numbers rather than missing ones.

Every test here corresponds to a defect where the profiler kept running, kept writing files
that parsed, and reported a result that looked complete. That is the property being defended:
each assertion checks not only that the number is right, but that the loss is declared.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest

from lineprofiler.accounting import Profiler, merge_run, render
from lineprofiler.accounting.analysis import analyse
from lineprofiler.accounting.identity import hostname
from lineprofiler.accounting.phase import PhaseStats, PhaseTree
from lineprofiler.accounting.profiler import MAX_PHASES
from lineprofiler.accounting.sampler import IoSnapshot, Sample, read_io_snapshot
from lineprofiler.accounting.snapshot import SnapshotWriter, new_run_id

# ── a failed counter read must not be differenced ───────────────────────────


def _io_series(rows: list[tuple[float, str, int, bool]]) -> list[Sample]:
    """(t, phase, cumulative_read_bytes, counters_readable) → samples."""
    return [
        Sample(t=t, phase=phase, rss=1_000, read_bytes=read, read_chars=read, io_ok=ok)
        for t, phase, read, ok in rows
    ]


def test_a_failed_counter_read_does_not_fabricate_cumulative_traffic() -> None:
    """The defect: a zero row differenced against the next real one billed a phase for the
    process's entire lifetime of I/O — 372 GB materialising on whichever phase was open."""
    samples = _io_series([
        (0.0, "train", 400_000_000_000, True),
        (1.0, "checkpoint", 0, False),          # /proc read failed; the zeros mean nothing
        (2.0, "train", 400_000_002_000, True),
    ])

    analysis = analyse(samples)

    assert analysis.totals.read_bytes == 0, "no interval spans the gap, so nothing is claimed"
    assert analysis.io_by_phase["checkpoint"].read_bytes == 0


def test_intervals_either_side_of_a_gap_are_both_discarded() -> None:
    samples = _io_series([
        (0.0, "a", 1_000, True),
        (1.0, "b", 0, False),
        (2.0, "c", 9_000, True),
        (3.0, "c", 9_500, True),
    ])

    analysis = analyse(samples)

    assert analysis.totals.read_bytes == 500, "only the fully-measured interval counts"
    assert analysis.io_gap_intervals == 2
    assert analysis.io_intervals == 3


def test_a_measured_run_reports_no_gaps() -> None:
    samples = _io_series([(0.0, "a", 0, True), (1.0, "a", 4_096, True)])

    analysis = analyse(samples)

    assert analysis.io_gap_intervals == 0
    assert analysis.totals.read_bytes == 4_096


def test_the_report_declares_dropped_intervals(tmp_path: Path) -> None:
    """A silent floor is the thing being prevented: the reader must be told."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train"):
        pass
    profiler.close()
    worker = next(iter((tmp_path / "workers").rglob("w_*.json")))
    rows = _io_series([
        (0.0, "train", 100_000, True),
        (1.0, "train", 0, False),
        (2.0, "train", 900_000, True),
    ])
    worker.with_suffix(".samples").write_text(
        "\n".join(
            json.dumps({
                "t": s.t, "phase": s.phase, "rss": s.rss,
                "read_bytes": s.read_bytes, "read_chars": s.read_chars,
                **({} if s.io_ok else {"io_ok": False}),
            })
            for s in rows
        ) + "\n",
        encoding="utf-8",
    )

    text = render(merge_run(tmp_path))

    assert "could not read" in text
    assert "lower bound" in text


def test_read_io_snapshot_reports_unavailability_rather_than_zero() -> None:
    assert read_io_snapshot(None).available is False
    assert IoSnapshot().available is True, "a real all-zero reading is still a reading"


def test_an_io_phase_records_nothing_when_a_boundary_read_failed(tmp_path: Path) -> None:
    """The exactly-measured block must stay exact: no reading is better than a fake one."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    if profiler.io_counters().is_empty():
        pytest.skip("no per-process I/O counters on this platform")

    original = profiler.io_counters
    calls = {"n": 0}

    def failing_first_read() -> IoSnapshot:
        calls["n"] += 1
        return IoSnapshot(available=False) if calls["n"] == 1 else original()

    profiler.io_counters = failing_first_read  # type: ignore[method-assign]
    with profiler.phase("checkpoint", io=True):
        (tmp_path / "payload").write_bytes(b"x" * 200_000)
    profiler.io_counters = original  # type: ignore[method-assign]
    profiler.close()

    counters = merge_run(tmp_path).tree[("checkpoint",)].counters
    assert not any(name.startswith("io_") for name in counters), counters


# ── a failing snapshot must not end flushing ────────────────────────────────


def test_a_raising_snapshot_does_not_stop_the_ones_after_it(tmp_path: Path) -> None:
    """The defect: _on_timer re-armed only on success, so one exception froze the worker
    file for the rest of the run — still valid JSON, hours out of date."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=0.05, sample_interval_s=None,
    )
    writer = profiler._writer  # noqa: SLF001
    assert writer is not None
    failures = {"n": 0}
    real_write = writer.write

    def fail_twice(tree: PhaseTree) -> bool:
        failures["n"] += 1
        if failures["n"] <= 2:
            raise OSError(28, "No space left on device")
        return real_write(tree)

    writer.write = fail_twice  # type: ignore[method-assign]
    with profiler.phase("train"):
        pass
    deadline = time.monotonic() + 5.0
    while failures["n"] < 4 and time.monotonic() < deadline:
        time.sleep(0.05)
    writer.write = real_write  # type: ignore[method-assign]
    profiler.close()

    assert failures["n"] >= 4, "flushing stopped after the first failure"
    assert merge_run(tmp_path).tree[("train",)].calls == 1


def test_the_writer_reports_a_failure_instead_of_raising(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path, role="learner", run_id=new_run_id())
    writer.path = tmp_path / "no-such-directory" / "w.json"

    assert writer.write({}) is False
    assert writer.write_failures == 1
    assert writer.last_error is not None


def test_construction_survives_an_unwritable_run_directory(tmp_path: Path) -> None:
    """The profiler must never be the reason a twelve-hour job dies."""
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory", encoding="utf-8")

    writer = SnapshotWriter(blocked, role="actor", run_id=new_run_id())

    assert writer.write_failures >= 1


def test_a_counter_added_during_a_merge_does_not_raise() -> None:
    """The defect: PhaseStats.merge iterated a live dict, and the owning thread inserting a
    new counter name mid-merge raised RuntimeError, killing the flush thread."""
    source = PhaseStats()
    for index in range(200):
        source.add_count(f"counter_{index}", 1)
    target = PhaseStats()
    started = threading.Event()

    def keep_adding() -> None:
        """Introduce new counter names steadily, which is what used to break the merge."""
        started.set()
        for index in range(1_000, 4_000):
            source.add_count(f"counter_{index}", 1)
            if index % 64 == 0:
                time.sleep(0)  # hand the GIL over mid-iteration

    writer = threading.Thread(target=keep_adding, daemon=True)
    writer.start()
    started.wait(timeout=2.0)
    while writer.is_alive():
        target.merge(source)  # must not raise "dictionary changed size during iteration"
    writer.join(timeout=5.0)

    assert target.counters["counter_0"] > 0


# ── attempts must not be merged together ────────────────────────────────────


def _one_attempt(run_dir: Path, phase: str, calls: int) -> None:
    """Run a whole attempt in a fresh process, so it gets its own run id."""
    code = "\n".join([
        "from lineprofiler.accounting import Profiler",
        f"p = Profiler(run_dir={str(run_dir)!r}, enabled=True,",
        "             snapshot_interval_s=None, sample_interval_s=None)",
        f"for _ in range({calls}):",
        f"    with p.phase({phase!r}):",
        "        pass",
        "p.close()",
    ])
    environment = {k: v for k, v in os.environ.items() if not k.startswith("LINEPROFILER_")}
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_a_second_attempt_supersedes_the_first(tmp_path: Path) -> None:
    """The defect: a requeued job merged the abandoned attempt's workers into the new run,
    inflating every total with nothing in the report to say so."""
    _one_attempt(tmp_path, "train", 3)
    time.sleep(1.05)  # run ids carry a one-second timestamp resolution
    _one_attempt(tmp_path, "train", 5)

    run = merge_run(tmp_path)

    assert run.tree[("train",)].calls == 5, "only the newest attempt counts"
    assert len(run.workers) == 1
    assert len(run.superseded) == 1
    assert "earlier attempt" in render(run)


def test_workers_of_one_attempt_are_not_split(tmp_path: Path) -> None:
    """Children inherit the run id through the environment, so a spawned worker joins its
    parent's attempt rather than starting a new one."""
    parent = Profiler(
        run_dir=tmp_path, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    child = Profiler(
        run_dir=tmp_path, role="actor", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    assert child.run_id == parent.run_id
    with parent.phase("learn"):
        pass
    with child.phase("act"):
        pass
    parent.close()
    child.close()

    run = merge_run(tmp_path)

    assert run.superseded == []
    assert len(run.workers) == 2


# ── a bad file must cost one worker, not the run ────────────────────────────


def test_a_structurally_invalid_worker_is_lost_alone(tmp_path: Path) -> None:
    """The defect: valid JSON with the wrong shape raised straight out of merge_run and
    aborted the whole report."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train"):
        pass
    profiler.close()
    workers = tmp_path / "workers"
    (workers / "w_bogus_1_deadbeef.json").write_text(
        json.dumps({"version": 99, "unexpected": "shape"}), encoding="utf-8",
    )

    run = merge_run(tmp_path)

    assert run.tree[("train",)].calls == 1, "the good worker still reports"
    assert len(run.unreadable) == 1
    assert "unreadable" in render(run)


def test_a_worker_with_a_corrupt_histogram_is_lost_alone(tmp_path: Path) -> None:
    workers = tmp_path / "workers"
    workers.mkdir(parents=True)
    (workers / "w_x_1_aaaaaaaa.json").write_text(
        json.dumps({
            "version": 1, "pid": 1, "role": "main",
            "started_at": 0.0, "written_at": 1.0,
            "phases": {"train": {
                "calls": 1, "wall_ns": 10, "cpu_ns": 5, "child_wall_ns": 0,
                "hist": {"99999": 1}, "counters": {},
            }},
        }),
        encoding="utf-8",
    )

    run = merge_run(tmp_path)

    assert run.workers == []
    assert len(run.unreadable) == 1


# ── multi-node identity ─────────────────────────────────────────────────────


def test_processes_are_counted_by_worker_not_by_pid(tmp_path: Path) -> None:
    """The defect: pid namespaces are per-node, so ranks on different nodes shared a pid and
    the header undercounted every multi-node run."""
    workers = tmp_path / "workers"
    workers.mkdir(parents=True)
    for index, host in enumerate(("node01", "node02", "node03")):
        (workers / f"w_run_4242_{index:08d}.json").write_text(
            json.dumps({
                "version": 1, "run_id": "run", "pid": 4242, "role": "actor",
                "started_at": 0.0, "written_at": 60.0,
                "placement": {"host": host, "rank": index},
                "phases": {"act": {
                    "calls": 1, "wall_ns": 1_000, "cpu_ns": 500, "child_wall_ns": 0,
                    "hist": {"80": 1}, "counters": {},
                }},
            }),
            encoding="utf-8",
        )

    run = merge_run(tmp_path)
    text = render(run)

    assert len(run.workers) == 3
    assert run.hosts == ["node01", "node02", "node03"]
    assert "Processes 3" in text
    assert "3 nodes" in text


def test_a_worker_records_the_host_it_ran_on(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train"):
        pass
    profiler.close()

    worker = merge_run(tmp_path).workers[0]

    assert worker.host not in ("", "?")
    assert worker.placement["host"] == worker.host


# ── scheduler signals ───────────────────────────────────────────────────────


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGUSR1", "SIGHUP"])
def test_a_scheduler_signal_flushes_before_the_process_dies(
    tmp_path: Path, signame: str,
) -> None:
    """Slurm sends SIGUSR1 ahead of preemption; its default disposition skips atexit, so
    everything since the last periodic flush used to be lost."""
    code = "\n".join([
        "import time",
        "from lineprofiler.accounting import Profiler",
        f"p = Profiler(run_dir={str(tmp_path)!r}, enabled=True,",
        "             snapshot_interval_s=None, sample_interval_s=None)",
        "with p.phase('work'):",
        "    pass",
        "print('ready', flush=True)",
        "time.sleep(30)",
    ])
    environment = {k: v for k, v in os.environ.items() if not k.startswith("LINEPROFILER_")}
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=environment,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    process.send_signal(getattr(signal, signame))
    process.wait(timeout=10)

    assert merge_run(tmp_path).tree[("work",)].calls == 1, f"{signame} lost the snapshot"


# ── bounded memory ──────────────────────────────────────────────────────────


def test_dynamic_phase_names_stop_growing_the_tree(tmp_path: Path) -> None:
    """The defect: phase(f"episode_{i}") grew the tree for the life of the process, and each
    node holds a dense 512-bucket histogram that is also rewritten into every snapshot."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with pytest.warns(RuntimeWarning, match="distinct paths"):
        for index in range(MAX_PHASES + 500):
            with profiler.phase(f"episode_{index}"):
                pass
    profiler.close()

    tree = merge_run(tmp_path).tree
    assert len(tree) <= MAX_PHASES + 1, f"tree grew to {len(tree)} paths"


def test_the_overflow_warning_names_the_culprit(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with pytest.warns(RuntimeWarning, match="phase name is built from data"):
        for index in range(MAX_PHASES + 2):
            with profiler.phase(f"step_{index}"):
                pass
    profiler.close()


def test_a_fixed_phase_name_never_warns(tmp_path: Path) -> None:
    """The cap must be far above anything hand-written instrumentation reaches."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(10_000):
            with profiler.phase("train"), profiler.phase("forward"):
                pass
    profiler.close()

    assert merge_run(tmp_path).tree[("train", "forward")].calls == 10_000


def test_reporting_without_samples_gives_the_same_phase_tree(tmp_path: Path) -> None:
    """The escape hatch for a large run: samples dominate memory, phases are what you need."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=0.05,
    )
    with profiler.phase("train"):
        time.sleep(0.2)
    profiler.close()

    full = merge_run(tmp_path)
    lean = merge_run(tmp_path, with_samples=False)

    assert lean.tree[("train",)].calls == full.tree[("train",)].calls
    assert full.samples_by_process(), "the fixture must actually have samples"
    assert lean.samples_by_process() == []
    assert "MEMORY" not in render(lean)


# ── multi-node file layout ──────────────────────────────────────────────────


def test_worker_files_are_sharded_by_host(tmp_path: Path) -> None:
    """One flat directory of two files per rank is a metadata hot spot on Lustre, and makes
    "which files came from the node that died" unanswerable."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train"):
        pass
    profiler.close()

    shards = [p for p in (tmp_path / "workers").iterdir() if p.is_dir()]
    assert len(shards) == 1
    assert shards[0].name == hostname()
    assert list(shards[0].glob("w_*.json"))


def test_the_merge_reads_every_shard(tmp_path: Path) -> None:
    workers = tmp_path / "workers"
    for index, host in enumerate(("node01", "node02")):
        shard = workers / host
        shard.mkdir(parents=True)
        (shard / f"w_run_{index}_aaaaaaaa.json").write_text(
            json.dumps({
                "version": 1, "run_id": "run", "pid": index, "role": "actor",
                "started_at": 0.0, "written_at": 10.0,
                "placement": {"host": host, "rank": index},
                "phases": {"act": {
                    "calls": 2, "wall_ns": 2_000, "cpu_ns": 1_000, "child_wall_ns": 0,
                    "hist": {"88": 2}, "counters": {},
                }},
            }),
            encoding="utf-8",
        )

    run = merge_run(tmp_path)

    assert run.tree[("act",)].calls == 4
    assert run.hosts == ["node01", "node02"]
