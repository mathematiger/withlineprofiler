"""Regression tests for failures that used to produce wrong numbers rather than missing ones.

Every test here corresponds to a defect where the profiler kept running, kept writing files
that parsed, and reported a result that looked complete. That is the property being defended:
each assertion checks not only that the number is right, but that the loss is declared.
"""

from __future__ import annotations

import gc
import json
import os
import signal
import subprocess
import sys
import threading
import time
import warnings
import weakref
from pathlib import Path

import pytest

from lineprofiler.accounting import Profiler, merge_run, render
from lineprofiler.accounting.analysis import NO_PHASE, analyse, analyse_processes
from lineprofiler.accounting.compare import PhaseDelta, _comparison_notes, _delta_row
from lineprofiler.accounting.identity import hostname
from lineprofiler.accounting.phasetree import PhaseStats, PhaseTree
from lineprofiler.accounting.profiler import (
    _NAME_SHAPE_WARN,
    ENV_RUN_DIR,
    MAX_PHASES,
    _resolve_run_dir,
)
from lineprofiler.accounting.report import (
    _counter_rows,
    _io_attribution_note,
    _io_phase_rows,
    _label,
    format_label,
)
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


# ── a share must not divide one counter layer by the other ──────────────────


def _mixed_layer_series() -> list[Sample]:
    """Cache-only reads while no phase is open, then a disk read inside one.

    Reproduces the reviewer's run: 110.3 MB served from page cache with no phase open, over
    816.0 KB that reached the device. Both figures are correct and both are printed correctly
    in the rows above the note; only the share mixed them.
    """
    return [
        Sample(t=0.0, phase="", rss=1_000, read_bytes=0, read_chars=0),
        Sample(t=1.0, phase="train", rss=1_000, read_bytes=0, read_chars=110_300_000),
        Sample(t=2.0, phase="train", rss=1_000, read_bytes=816_000, read_chars=111_116_000),
    ]


def test_unattributed_share_never_exceeds_one() -> None:
    """The defect: the two ``or`` fallbacks resolved independently, so page-cache chars were
    divided by disk bytes and the report printed ``13845% of reads``."""
    analysis = analyse(_mixed_layer_series())

    assert analysis.totals.read_bytes == 816_000, "the block layer is unchanged"
    assert analysis.io_by_phase[NO_PHASE].read_chars == 110_300_000
    assert analysis.io_by_phase[NO_PHASE].read_bytes == 0, "none of it reached the device"
    assert 0.0 <= analysis.unattributed_read_share <= 1.0
    assert analysis.unattributed_read_share == pytest.approx(110_300_000 / 111_116_000)


def test_unattributed_share_falls_back_to_the_block_layer_together() -> None:
    """Without a syscall layer both operands drop to bytes — never one of each."""
    samples = [
        Sample(t=0.0, phase="", rss=1_000, read_bytes=0),
        Sample(t=1.0, phase="train", rss=1_000, read_bytes=4_000),
        Sample(t=2.0, phase="train", rss=1_000, read_bytes=10_000),
    ]

    analysis = analyse(samples)

    assert analysis.unattributed_read_share == pytest.approx(4_000 / 10_000)


def test_the_report_names_the_layer_its_share_is_measured_in() -> None:
    """A bare percentage over two layers is unreadable; the note has to say which one."""
    note = "\n".join(_io_attribution_note(analyse(_mixed_layer_series())))

    assert "syscall layer" in note
    assert "13845%" not in note


# ── a truncated phase label must not read as a real phase ───────────────────


def test_a_long_phase_label_is_marked_and_keeps_its_leaf() -> None:
    """The defect: ``phase[-26:]`` printed ``train_step/forward_backward`` as
    ``rain_step/forward_backward`` — a name the reader can neither find nor grep."""
    samples = [
        Sample(t=0.0, phase="train_step/forward_backward", rss=1_000, read_bytes=0),
        Sample(t=1.0, phase="train_step/forward_backward", rss=1_000, read_bytes=1_300_000),
        Sample(t=2.0, phase="train_step/forward_backward", rss=1_000, read_bytes=1_300_000),
    ]

    row = _io_phase_rows(analyse(samples))[1]

    assert row.startswith("  …in_step/forward_backward"), "truncation is marked, tail kept"
    assert "rain_step" not in row, "an unmarked cut invents a phase that does not exist"
    assert row[27] == " ", "the label can never abut the column beside it"
    assert row[28] == "r"


def test_a_label_short_enough_to_fit_is_left_alone() -> None:
    assert format_label("train_step", 25) == "train_step"
    assert format_label("x" * 25, 25) == "x" * 25


def test_label_truncation_keeps_the_leaf_not_the_head() -> None:
    """``_label`` used to cut with ``[:27]``, discarding the end that identifies the phase."""
    label = _label(("optimisation_step", "forward_backward"))

    assert label.endswith("forward_backward"), "the leaf survives"
    assert label.startswith("…")
    assert len(label) <= 27


def test_the_comparison_table_marks_a_truncated_phase_too() -> None:
    """The same defect lived in ``compare.py``, where ``[-27:]`` turned
    ``iteration/checkpoint_to_object_store`` into ``/checkpoint_to_object_store`` — a label
    that reads as a top-level phase whose name begins with a slash."""
    delta = PhaseDelta(
        phase="iteration/checkpoint_to_object_store",
        calls_a=2, calls_b=2, per_call_a=1_000, per_call_b=1_000,
        p50_a=1_000, p50_b=1_000,
    )

    row = _delta_row(delta)

    assert row.startswith("…"), "truncation is marked"
    assert not row.startswith("/"), "an unmarked cut invents a top-level phase"
    assert row[27] == " ", "the label can never abut the column beside it"


def test_the_comparison_tail_note_marks_a_truncated_phase_too() -> None:
    """The tail-change note used ``[-24:]`` in a 24-wide field: unmarked, and no column gap."""
    delta = PhaseDelta(
        phase="iteration/checkpoint_to_object_store",
        calls_a=40, calls_b=40, per_call_a=1_000, per_call_b=2_000,
        p50_a=1_000, p50_b=1_000,
    )
    assert delta.tail_moved, "the fixture must reach the tail note at all"

    note = "\n".join(_comparison_notes([delta]))

    assert "…kpoint_to_object_store mean" in note, "marked, and the column gap survives"


# ── one run must not scatter across per-worker working directories ──────────


def test_a_relative_run_dir_is_resolved_before_it_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect: ``Path("profile")`` was exported verbatim through LINEPROFILER_RUN_DIR, so
    a child with its own working directory wrote its worker file somewhere else entirely.
    One run merged as several short ones, each missing most of its workers."""
    monkeypatch.delenv("SLURM_SUBMIT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_run_dir("profile")

    assert resolved.is_absolute()
    assert resolved == tmp_path / "profile"


def test_a_relative_run_dir_ignores_the_submit_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``$SLURM_SUBMIT_DIR`` is whatever launched the job, not where the user works.

    Under Open OnDemand it is the dashboard's own installation directory — observed as
    ``/var/www/ood/apps/sys/dashboard`` — which is typically not writable. Resolving a
    relative ``run_dir`` there would relocate the user's output somewhere less predictable
    than the working directory they typed it against.
    """
    monkeypatch.setenv("SLURM_SUBMIT_DIR", "/var/www/ood/apps/sys/dashboard")
    monkeypatch.chdir(tmp_path)

    assert _resolve_run_dir("profile") == tmp_path / "profile"


def test_an_absolute_run_dir_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    assert _resolve_run_dir(tmp_path) == tmp_path


def test_a_child_in_another_directory_joins_the_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the resolution is actually for: two processes, two working directories, one run."""
    monkeypatch.delenv("SLURM_SUBMIT_DIR", raising=False)
    monkeypatch.delenv(ENV_RUN_DIR, raising=False)
    parent_cwd, child_cwd = tmp_path / "rank0", tmp_path / "rank1"
    parent_cwd.mkdir()
    child_cwd.mkdir()

    monkeypatch.chdir(parent_cwd)
    parent = Profiler(
        run_dir="profile", role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    monkeypatch.chdir(child_cwd)
    child = Profiler(
        role="actor", enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    assert child.run_dir == parent.run_dir == parent_cwd / "profile"
    parent.close()
    child.close()


def test_a_fast_counter_does_not_collide_with_its_rate() -> None:
    """The defect: ``{rate:>12,.1f}`` exactly fills its field at eight figures, so the count
    ran straight into it and ``64`` at ``19,161,676.6/s`` printed as ``6419,161,676.6/s``."""
    row = _counter_rows({"samples": 64}, wall_ns=3_340)[0]

    assert "6419,161,676.6" not in row
    assert "64 " in row, "the count is separated from the rate by a literal space"
    assert "/s " in row, "and the rate from the per-each figure"


def test_a_counter_number_is_never_truncated_to_fit() -> None:
    """Truncating a number prints a wrong one. An overflowing field pushes the row right."""
    row = _counter_rows({"steps": 12_345_678_901}, wall_ns=1_000_000_000)[0]

    assert "12,345,678,901" in row


# ── a phase name built from data must not degrade the report silently ───────


def test_strict_names_rejects_a_name_built_from_data(tmp_path: Path) -> None:
    """``count()`` raises on a float; a generated phase name was the more damaging mistake
    and had no equivalent protection — it degrades the report rather than raising."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, strict_names=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        with profiler.phase("episode_1"):
            pass
        with pytest.raises(ValueError, match="built from data"), profiler.phase("episode_2"):
            pass
    finally:
        profiler.close()


def test_strict_names_allows_a_fixed_vocabulary_containing_digits(tmp_path: Path) -> None:
    """One name in isolation says nothing: ``conv2d`` and ``resnet50`` are good names.
    What gives a generated name away is repetition of a *shape*."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, strict_names=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for name in ("conv2d", "resnet50", "fp16_cast", "train_step"):
            with profiler.phase(name):
                pass
    finally:
        profiler.close()

    assert set(merge_run(tmp_path).tree) >= {("conv2d",), ("resnet50",), ("fp16_cast",)}


def test_generated_names_warn_before_the_tree_folds(tmp_path: Path) -> None:
    """The warning has to arrive while the report is still readable — MAX_PHASES is far too
    late, because by then the run is already unusable."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for index in range(_NAME_SHAPE_WARN + 40):
                with profiler.phase(f"episode_{index}"):
                    pass
    finally:
        profiler.close()

    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(messages) == 1, "warned once, not once per name"
    assert "episode_#" in messages[0]
    assert _NAME_SHAPE_WARN < MAX_PHASES, "the warning must precede the fold"


def test_a_fixed_vocabulary_never_warns(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(500):
                with profiler.phase("train_step"), profiler.phase("forward"):
                    pass
    finally:
        profiler.close()

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ── a sampled phase must never read as a measured one ───────────────────────


def test_a_sampled_phase_estimates_the_unsampled_total(tmp_path: Path) -> None:
    """The estimate has to be right, or the option is worthless."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for _ in range(1_000):
            with profiler.phase("sampled", sample=0.01):
                pass
            with profiler.phase("measured"):
                pass
    finally:
        profiler.close()

    tree = profiler.merged_tree()
    assert tree[("sampled",)].calls == 1_000, "10 measured entries scaled by a stride of 100"
    assert tree[("measured",)].calls == 1_000
    assert tree[("sampled",)].hist.count == 1_000


def test_a_sampled_phase_is_marked_as_an_estimate(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for _ in range(200):
            with profiler.phase("sampled", sample=0.01):
                pass
            with profiler.phase("measured"):
                pass
    finally:
        profiler.close()

    tree = profiler.merged_tree()
    assert tree[("sampled",)].sample_stride == 100
    assert tree[("measured",)].sample_stride == 0, "measurement must stay distinguishable"


def test_merging_a_sampled_node_into_a_measured_one_marks_the_result() -> None:
    """Otherwise a merged total presents partly-estimated numbers as measured."""
    measured = PhaseStats(calls=10, wall_ns=1_000)
    sampled = PhaseStats(calls=100, wall_ns=10_000, sample_stride=100)

    measured.merge(sampled)

    assert measured.sample_stride == 100


def test_nothing_under_a_skipped_sampled_entry_is_recorded(tmp_path: Path) -> None:
    """The subtle one. If children kept recording at full rate under a parent measured at one
    in n, the tree would mix two rates and every share derived from it would be wrong — a
    plausible wrong number rather than an obvious one."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for _ in range(100):
            with profiler.phase("outer", sample=0.1), profiler.phase("inner"):
                profiler.count("units", 1)
    finally:
        profiler.close()

    tree = profiler.merged_tree()
    assert tree[("outer",)].calls == 100
    assert tree[("outer", "inner")].calls == 100, (
        "the child must be scaled with its parent, not counted at full rate"
    )
    assert tree[("outer", "inner")].counters == {"units": 100}
    assert tree[("outer", "inner")].sample_stride == 10, "the child is an estimate too"


def test_counters_outside_a_selected_entry_are_not_recorded(tmp_path: Path) -> None:
    """count() inside a skipped entry must be dropped, not attributed to the ancestor."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        with profiler.phase("root"):
            for _ in range(100):
                with profiler.phase("sampled", sample=0.1):
                    profiler.count("units", 1)
    finally:
        profiler.close()

    tree = profiler.merged_tree()
    assert tree[("root", "sampled")].counters == {"units": 100}
    assert "units" not in tree[("root",)].counters, "a skipped body must not bill its parent"


def test_the_report_marks_and_explains_a_sampled_row(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for _ in range(200):
            with profiler.phase("uct_search", sample=0.01):
                time.sleep(0.0001)
    finally:
        profiler.close()

    text = render(merge_run(tmp_path))

    assert "~uct_search" in text
    assert "estimated from a sample, not measured" in text
    assert "1 entry in 100" in text


def test_an_unsampled_run_says_nothing_about_sampling(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        with profiler.phase("train_step"):
            time.sleep(0.001)
    finally:
        profiler.close()

    text = render(merge_run(tmp_path))

    assert "estimated from a sample" not in text
    assert "~" not in text


def test_an_impossible_sample_rate_raises(tmp_path: Path) -> None:
    """``sample=0`` means "measure nothing" — a mistake worth hearing about immediately."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="sample must be in"):
                profiler.phase("x", sample=bad)
    finally:
        profiler.close()


def test_sample_stride_survives_the_snapshot_round_trip(tmp_path: Path) -> None:
    """An estimate that reads back as a measurement is the failure being prevented."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        for _ in range(200):
            with profiler.phase("sampled", sample=0.01):
                pass
    finally:
        profiler.close()

    assert merge_run(tmp_path).tree[("sampled",)].sample_stride == 100


def test_a_worker_file_without_sample_stride_reads_as_measured(tmp_path: Path) -> None:
    """0.3.0 worker files predate the field, and everything in them really was measured."""
    stats = PhaseStats.from_dict({
        "calls": 4, "wall_ns": 40, "cpu_ns": 20, "child_wall_ns": 0,
        "hist": {"8": 4}, "counters": {},
    })

    assert stats.sample_stride == 0


# ── an enabled profiler must not outlive itself ─────────────────────────────
#
# close() used to stop the threads and write the final snapshot but leave every process-global
# hook it installed in place: the atexit registration, the three scheduler signal handlers, and
# the os.register_at_fork callbacks. A process that merely constructed and closed a profiler was
# permanently altered, and the damage surfaced far from the cause — in the reported case, two
# in-process profiler tests left the interpreter unable to terminate its own forked children,
# and the failures appeared in an unrelated file several hundred tests away.


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGUSR1", "SIGHUP"])
def test_close_restores_the_signal_handler_it_replaced(tmp_path: Path, signame: str) -> None:
    """A closed profiler must leave the process's signal dispositions as it found them."""
    signum = getattr(signal, signame)
    before = signal.getsignal(signum)

    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    assert signal.getsignal(signum) is not before, "the profiler did not chain the signal"
    profiler.close()

    assert signal.getsignal(signum) is before


def test_close_does_not_clobber_a_handler_installed_after_the_profiler(tmp_path: Path) -> None:
    """The profiler is no longer top of the chain, so restoring would delete the host's handler.

    Leaving ours installed is the safe side of the trade: it still chains correctly to whatever
    was there, it is merely no longer removable.
    """
    original = signal.getsignal(signal.SIGUSR1)
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def host_handler(number: int, frame: object) -> None:  # pragma: no cover - never raised
        pass

    signal.signal(signal.SIGUSR1, host_handler)
    try:
        profiler.close()
        assert signal.getsignal(signal.SIGUSR1) is host_handler
    finally:
        signal.signal(signal.SIGUSR1, original)


def test_closing_twice_restores_once_and_does_not_raise(tmp_path: Path) -> None:
    """Idempotence matters: close() is reachable from atexit, a signal and an explicit call."""
    before = signal.getsignal(signal.SIGTERM)
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    profiler.close()
    profiler.close()

    assert signal.getsignal(signal.SIGTERM) is before


def test_a_closed_profiler_can_be_garbage_collected(tmp_path: Path) -> None:
    """The fork callbacks used to be three bound methods per profiler, and
    ``os.register_at_fork`` has no counterpart that unregisters — so every enabled profiler
    was immortal, holding its phase trees, thread states and writer for the life of the
    interpreter. A suite that constructs one per test paid for all of them at once.
    """
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    ref = weakref.ref(profiler)
    profiler.close()
    del profiler
    gc.collect()

    assert ref() is None


def test_close_unregisters_the_atexit_hook(tmp_path: Path) -> None:
    """Asserted through reachability rather than ``atexit._ncallbacks()``, which does not
    shrink on unregister and so cannot tell a removed callback from a retained one."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    ref = weakref.ref(profiler)
    profiler.close()
    del profiler
    gc.collect()

    # atexit holds the bound method strongly; if it were still registered the object would live.
    assert ref() is None


def test_closing_out_of_order_still_leaves_the_process_as_it_was(tmp_path: Path) -> None:
    """Handlers chain, so the profiler currently installed is the last one constructed — but
    closing order need not match construction order, and a parent closed before the child it
    made is ordinary code. The parent is no longer on top, so it cannot simply restore; it
    hands its predecessor to the child instead. Without that splice the parent's handler stayed
    in the process for good, which is how this suite leaked one per test.
    """
    before = signal.getsignal(signal.SIGTERM)
    parent = Profiler(
        run_dir=tmp_path, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    child = Profiler(
        run_dir=tmp_path, role="actor", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )

    parent.close()
    assert signal.getsignal(signal.SIGTERM) is not before, "the child is still live"
    child.close()

    assert signal.getsignal(signal.SIGTERM) is before


def test_a_signal_still_flushes_while_an_out_of_order_close_is_pending(
    tmp_path: Path,
) -> None:
    """The splice rewrites a restore target, not a live handler, so the chain must still run
    end to end while the closed profiler is technically still installed.

    A sentinel is installed at the bottom of the chain first, both to prove the chain reaches
    all the way down and because the real bottom is ``SIG_DFL``, which would kill the test
    process rather than return.
    """
    original = signal.getsignal(signal.SIGUSR1)
    reached: list[str] = []

    def sentinel(number: int, frame: object) -> None:
        reached.append("bottom")

    signal.signal(signal.SIGUSR1, sentinel)
    try:
        parent = Profiler(
            run_dir=tmp_path / "p", enabled=True,
            snapshot_interval_s=None, sample_interval_s=None,
        )
        child = Profiler(
            run_dir=tmp_path / "c", enabled=True,
            snapshot_interval_s=None, sample_interval_s=None,
        )
        with child.phase("work"):
            pass
        parent.close()

        handler = signal.getsignal(signal.SIGUSR1)
        assert callable(handler)
        handler(int(signal.SIGUSR1), None)  # walk the chain without raising a real signal

        assert reached == ["bottom"], "the chain did not reach the handler underneath"
        assert merge_run(tmp_path / "c").tree[("work",)].calls == 1

        child.close()
        assert signal.getsignal(signal.SIGUSR1) is sentinel
    finally:
        signal.signal(signal.SIGUSR1, original)


def test_a_worker_file_without_hardware_still_merges(tmp_path: Path) -> None:
    """Every run recorded before the inventory existed must stay readable."""
    workers = tmp_path / "workers"
    workers.mkdir(parents=True)
    (workers / "w_1_a.json").write_text(
        json.dumps({
            "version": 1, "pid": 1, "role": "actor",
            "started_at": 1.0, "written_at": 2.0,
            "phases": {"work": {"calls": 1, "wall_ns": 1000, "cpu_ns": 500,
                                "child_wall_ns": 0, "hist": {}, "counters": {}}},
        }),
        encoding="utf-8",
    )

    run = merge_run(tmp_path)

    assert run.workers[0].hardware == {}
    assert run.hardware_by_host == {}
    assert "RESOURCES" not in render(run)


def test_a_sample_row_without_cpu_percent_reads_as_unmeasured(tmp_path: Path) -> None:
    """The sentinel, not zero: an old row must not report a busy process as idle."""
    workers = tmp_path / "workers"
    workers.mkdir(parents=True)
    (workers / "w_1_a.json").write_text(
        json.dumps({
            "version": 1, "pid": 1, "role": "actor",
            "started_at": 1.0, "written_at": 2.0,
            "phases": {"work": {"calls": 1, "wall_ns": 1000, "cpu_ns": 500,
                                "child_wall_ns": 0, "hist": {}, "counters": {}}},
        }),
        encoding="utf-8",
    )
    (workers / "w_1_a.samples").write_text(
        "\n".join(json.dumps({"t": float(i), "phase": "work", "rss": 1000}) for i in range(3)),
        encoding="utf-8",
    )

    analysis = analyse_processes(merge_run(tmp_path).samples_by_process())

    assert not analysis.cpu.measured
    assert analysis.memory.peak_rss == 1000
