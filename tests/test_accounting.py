"""Tests for the accounting layer (Phase 1: core accounting, single process)."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from lineprofiler import accounting
from lineprofiler.accounting import DurationHistogram, PhaseStats, Profiler, merge_run
from lineprofiler.accounting.histogram import BUCKET_COUNT, bucket_index, bucket_lower_ns
from lineprofiler.accounting.phasetree import PhaseTree, merge_trees
from lineprofiler.accounting.report import render

# ── histogram ───────────────────────────────────────────────────────────────


def test_bucket_index_is_monotone_and_in_range() -> None:
    previous = -1
    for value in [0, 1, 5, 15, 16, 17, 31, 1_000, 999_999, 10**9, 10**12, 10**15]:
        index = bucket_index(value)
        assert 0 <= index < BUCKET_COUNT
        assert index >= previous
        previous = index


def test_bucket_lower_bound_never_exceeds_its_values() -> None:
    for value in [1, 15, 16, 100, 12_345, 10**9, 3 * 10**12]:
        assert bucket_lower_ns(bucket_index(value)) <= value


def test_small_durations_are_exact() -> None:
    for value in range(16):
        assert bucket_index(value) == value


def test_merge_is_commutative_and_associative() -> None:
    def build(values: list[int]) -> DurationHistogram:
        histogram = DurationHistogram()
        for value in values:
            histogram.observe(value)
        return histogram

    a, b, c = [10, 20, 30], [40, 50], [60, 70, 80, 90]

    left = build(a)
    left.merge(build(b))
    left.merge(build(c))

    right = build(c)
    right.merge(build(a))
    right.merge(build(b))

    assert left.buckets == right.buckets
    assert left.count == right.count == 9


def test_merging_empty_histogram_is_the_identity() -> None:
    histogram = DurationHistogram()
    for value in [1, 2, 3]:
        histogram.observe(value)
    before = list(histogram.buckets)

    histogram.merge(DurationHistogram())

    assert histogram.buckets == before
    assert histogram.count == 3


def test_quantiles_are_within_one_bucket_width_on_a_lognormal() -> None:
    rng = random.Random(7)
    samples = [int(math.exp(rng.gauss(12.0, 1.5))) for _ in range(10_000)]
    histogram = DurationHistogram()
    for value in samples:
        histogram.observe(value)
    samples.sort()

    for q in (0.5, 0.95, 0.99):
        exact = samples[int(q * len(samples)) - 1]
        estimate = histogram.quantile(q)
        assert abs(estimate - exact) / exact < 0.10, f"q={q} exact={exact} est={estimate}"


def test_sparse_round_trip_is_exact() -> None:
    histogram = DurationHistogram()
    for value in [3, 3, 4_000, 10**9]:
        histogram.observe(value)

    restored = DurationHistogram.from_sparse(histogram.to_sparse())

    assert restored.buckets == histogram.buckets
    assert restored.count == histogram.count


# ── phase statistics ────────────────────────────────────────────────────────


def test_self_and_wait_are_never_negative() -> None:
    stats = PhaseStats(calls=1, wall_ns=100, cpu_ns=250, child_wall_ns=400)
    assert stats.self_ns == 0
    assert stats.wait_ns == 0


def test_merge_trees_keeps_paths_present_in_only_one_tree() -> None:
    left: PhaseTree = {("a",): PhaseStats(calls=1, wall_ns=10)}
    right: PhaseTree = {
        ("a",): PhaseStats(calls=2, wall_ns=20),
        ("b",): PhaseStats(calls=5, wall_ns=50),
    }

    merge_trees(left, right)

    assert left[("a",)].calls == 3
    assert left[("a",)].wall_ns == 30
    assert left[("b",)].calls == 5


def test_merge_trees_does_not_alias_the_source() -> None:
    """A node inserted into the target must not still be the source's own object."""
    source: PhaseTree = {("only_here",): PhaseStats(calls=1, wall_ns=100)}
    first: PhaseTree = {}
    merge_trees(first, source)
    merge_trees(first, source)

    assert first[("only_here",)].wall_ns == 200
    assert source[("only_here",)].wall_ns == 100, "merging must leave the source untouched"


# ── phases ──────────────────────────────────────────────────────────────────


def test_known_sleep_durations_are_reproduced_within_five_percent(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("outer"):
        time.sleep(0.05)
        with profiler.phase("inner"):
            time.sleep(0.10)
    tree = profiler.merged_tree()

    outer = tree[("outer",)]
    inner = tree[("outer", "inner")]

    assert inner.wall_ns == pytest.approx(0.10e9, rel=0.05)
    assert outer.wall_ns == pytest.approx(0.15e9, rel=0.05)
    assert outer.child_wall_ns == pytest.approx(0.10e9, rel=0.05)
    assert outer.self_ns == pytest.approx(0.05e9, rel=0.05)


def test_sleeping_phase_is_almost_entirely_wait(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("blocked"):
        time.sleep(0.05)

    stats = profiler.merged_tree()[("blocked",)]
    assert stats.wait_ns / stats.wall_ns > 0.9


def test_exception_inside_a_phase_still_records_it(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with pytest.raises(ValueError), profiler.phase("boom"):
        raise ValueError("expected")

    tree = profiler.merged_tree()
    assert tree[("boom",)].calls == 1


def test_recursion_does_not_double_count_the_parent(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def descend(depth: int) -> None:
        with profiler.phase("rec"):
            time.sleep(0.01)
            if depth:
                descend(depth - 1)

    descend(2)
    tree = profiler.merged_tree()

    for path, stats in tree.items():
        assert stats.self_ns >= 0, path
    outermost = tree[("rec",)]
    assert outermost.self_ns == pytest.approx(0.01e9, rel=0.5)
    assert outermost.wall_ns == pytest.approx(0.03e9, rel=0.3)


def test_depth_beyond_the_cap_folds_into_its_ancestor(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def descend(depth: int) -> None:
        if not depth:
            return
        with profiler.phase("deep"):
            descend(depth - 1)

    descend(80)
    tree = profiler.merged_tree()

    assert max(len(path) for path in tree) <= 32


def test_phase_stacks_are_thread_local(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        with profiler.phase(name):
            barrier.wait()
            time.sleep(0.02)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tree = profiler.merged_tree()
    assert ("left",) in tree
    assert ("right",) in tree
    assert tree[("left",)].child_wall_ns == 0
    assert tree[("right",)].child_wall_ns == 0


def test_abandoned_generator_body_still_closes_its_phase(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def produce() -> object:
        with profiler.phase("gen"):
            yield 1
            yield 2

    generator = produce()
    next(generator)  # type: ignore[call-overload]
    generator.close()  # type: ignore[attr-defined]

    with profiler.phase("after"):
        pass

    tree = profiler.merged_tree()
    assert tree[("gen",)].calls == 1
    assert ("after",) in tree, "the abandoned phase must not still be on the stack"


# ── counters ────────────────────────────────────────────────────────────────


def test_counting_outside_any_phase_lands_on_the_root(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    profiler.count("orphan", 3)

    assert profiler.merged_tree()[()].counters == {"orphan": 3}


def test_counters_attribute_to_the_enclosing_phase(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("mcts"):
        profiler.count("mcts_simulations", 64)
        profiler.count("mcts_simulations")

    assert profiler.merged_tree()[("mcts",)].counters == {"mcts_simulations": 65}


def test_float_counts_are_rejected(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with pytest.raises(TypeError):
        profiler.count("steps", 1.5)  # type: ignore[arg-type]


# ── disabled profiler ───────────────────────────────────────────────────────


def test_disabled_profiler_creates_no_files_and_no_threads(tmp_path: Path) -> None:
    run_dir = tmp_path / "profile"
    before = threading.active_count()

    profiler = Profiler(run_dir=run_dir, enabled=False)
    with profiler.phase("noop"):
        profiler.count("ignored", 5)
    profiler.snapshot()
    profiler.close()

    assert not run_dir.exists()
    assert threading.active_count() == before
    assert profiler.merged_tree() == {}


def test_a_disabled_profiler_opens_no_process_handle(tmp_path: Path) -> None:
    """``open_process()`` constructs a psutil.Process and reads /proc. It used to run
    unconditionally, above the enabled check, so "allocate nothing" was not quite true."""
    profiler = Profiler(run_dir=tmp_path, enabled=False)

    assert profiler._process is None  # noqa: SLF001
    assert profiler.io_counters().is_empty(), "and it still answers rather than raising"


def test_environment_variable_is_read_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LINEPROFILER_PROFILE", "1")
    profiler = Profiler(
        run_dir=tmp_path, snapshot_interval_s=None, sample_interval_s=None,
    )
    assert profiler.enabled is True

    monkeypatch.delenv("LINEPROFILER_PROFILE")
    assert profiler.enabled is True, "enabled must not be re-read per call"
    profiler.close()


def test_explicit_enabled_overrides_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEPROFILER_PROFILE", "1")
    assert Profiler(enabled=False).enabled is False


def test_falsy_environment_values_keep_it_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("", "0", "false"):
        monkeypatch.setenv("LINEPROFILER_PROFILE", value)
        assert Profiler().enabled is False


# ── snapshots and merge ─────────────────────────────────────────────────────


def test_snapshot_round_trip_preserves_phases_and_counters(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train_step"):
        profiler.count("train_samples", 128)
        time.sleep(0.01)
    profiler.snapshot()

    run = merge_run(tmp_path)

    assert run.tree[("train_step",)].counters == {"train_samples": 128}
    assert run.tree[("train_step",)].calls == 1
    assert run.tree[("train_step",)].hist.count == 1
    assert run.unreadable == []


def test_two_snapshots_in_quick_succession_both_leave_a_valid_file(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("a"):
        pass
    profiler.snapshot()
    with profiler.phase("a"):
        pass
    profiler.snapshot()

    run = merge_run(tmp_path)
    assert run.tree[("a",)].calls == 2
    assert len(run.workers) == 1


def test_unreadable_worker_file_is_reported_not_raised(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("a"):
        pass
    profiler.snapshot()
    (tmp_path / "workers" / "w_999_deadbeef.json").write_text("{truncated", encoding="utf-8")

    run = merge_run(tmp_path)

    assert len(run.unreadable) == 1
    assert run.tree[("a",)].calls == 1


def test_merge_is_order_independent(tmp_path: Path) -> None:
    workers = tmp_path / "workers"
    workers.mkdir(parents=True)
    for pid, calls in ((1, 3), (2, 7)):
        payload = {
            "version": 1,
            "pid": pid,
            "started_at": 0.0,
            "written_at": 1.0,
            "phases": {
                "mcts": {
                    "calls": calls,
                    "wall_ns": calls * 1000,
                    "cpu_ns": 0,
                    "child_wall_ns": 0,
                    "hist": {},
                    "counters": {"sims": calls},
                },
            },
        }
        (workers / f"w_{pid}_aaaaaaaa.json").write_text(json.dumps(payload), encoding="utf-8")

    run = merge_run(tmp_path)

    assert run.tree[("mcts",)].calls == 10
    assert run.tree[("mcts",)].counters == {"sims": 10}
    assert run.imbalance == pytest.approx(1.4)


def test_report_renders_a_merged_run(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("self_play"), profiler.phase("mcts"):
        profiler.count("mcts_simulations", 64)
        time.sleep(0.01)
    profiler.snapshot()

    text = render(merge_run(tmp_path))

    assert "self_play" in text
    assert "mcts_simulations" in text
    assert "imbalance" in text
    assert "MAIN" in text, "the default role gets its own block"


def test_report_of_an_empty_run_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "workers").mkdir(parents=True)
    assert "Processes 0" in render(merge_run(tmp_path))


# ── crash resilience ────────────────────────────────────────────────────────


_SIGTERM_SCRIPT = """
    import os, signal, time
    from lineprofiler.accounting import Profiler

    profiler = Profiler(run_dir={run_dir!r}, enabled=True, snapshot_interval_s=0.2)
    with profiler.phase("work"):
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)
    """


def test_sigterm_leaves_a_parseable_snapshot(tmp_path: Path) -> None:
    script = textwrap.dedent(_SIGTERM_SCRIPT).format(run_dir=str(tmp_path))
    killed_at = time.time()
    subprocess.run([sys.executable, "-c", script], timeout=30, check=False)

    run = merge_run(tmp_path)

    assert run.unreadable == []
    assert len(run.workers) == 1
    assert run.workers[0].written_at >= killed_at
    assert time.time() - run.workers[0].written_at < 5.0


# ── the ambient (installed) profiler ────────────────────────────────────────


def test_module_level_phase_is_a_no_op_with_nothing_installed() -> None:
    """Library code can be instrumented without knowing whether it is being profiled."""
    accounting.uninstall_profiler()

    with accounting.phase("unobserved"):
        accounting.count("units", 5)

    assert accounting.installed_profiler() is None
    assert accounting.current() == ""


def test_module_level_phase_records_on_the_installed_profiler(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        assert accounting.installed_profiler() is profiler
        with accounting.phase("iteration"), accounting.phase("mcts"):
            accounting.count("simulations", 64)
            assert accounting.current() == "iteration/mcts"
    finally:
        profiler.close()

    tree = profiler.merged_tree()
    assert tree[("iteration", "mcts")].calls == 1
    assert tree[("iteration", "mcts")].counters == {"simulations": 64}


def test_closing_uninstalls_so_a_dead_profiler_is_never_resolvable(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    profiler.close()

    assert accounting.installed_profiler() is None
    with accounting.phase("after_close"):
        accounting.count("ignored", 1)

    assert ("after_close",) not in merge_run(tmp_path).tree


def test_installing_a_second_profiler_warns(tmp_path: Path) -> None:
    first = Profiler(
        run_dir=tmp_path / "a", enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        with pytest.warns(RuntimeWarning, match="already installed"):
            second = Profiler(
                run_dir=tmp_path / "b", enabled=True, install=True,
                snapshot_interval_s=None, sample_interval_s=None,
            )
        assert accounting.installed_profiler() is second
        second.close()
    finally:
        first.close()
        accounting.uninstall_profiler()


def test_an_explicitly_disabled_profiler_still_installs_as_a_no_op(tmp_path: Path) -> None:
    """``enabled=False`` must not leave module-level calls resolving to a stale profiler."""
    profiler = Profiler(run_dir=tmp_path, enabled=False, install=True)
    try:
        with accounting.phase("noop"):
            accounting.count("ignored", 1)
        assert profiler.merged_tree() == {}
    finally:
        profiler.close()
        accounting.uninstall_profiler()


# ── live export ─────────────────────────────────────────────────────────────


def test_two_successive_delta_reads_sum_to_the_cumulative_tree(tmp_path: Path) -> None:
    """The property every hand-rolled delta cache is trying to have."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    for _ in range(3):
        with profiler.phase("step"):
            profiler.count("units", 2)
    first = profiler.deltas()
    for _ in range(4):
        with profiler.phase("step"):
            profiler.count("units", 2)
    second = profiler.deltas()
    profiler.close()

    assert first[("step",)].calls == 3
    assert second[("step",)].calls == 4
    total = profiler.merged_tree()[("step",)]
    assert first[("step",)].calls + second[("step",)].calls == total.calls
    assert first[("step",)].counters["units"] + second[("step",)].counters["units"] == 14
    assert first[("step",)].wall_ns + second[("step",)].wall_ns == total.wall_ns


def test_a_phase_idle_over_the_interval_is_absent_from_the_deltas(tmp_path: Path) -> None:
    """Absent, not present at zero: an exporter must not publish a flat line as activity."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("once"):
        pass
    profiler.deltas()
    with profiler.phase("again"):
        pass
    deltas = profiler.deltas()
    profiler.close()

    assert ("again",) in deltas
    assert ("once",) not in deltas


def test_delta_quantiles_describe_the_interval_not_the_run(tmp_path: Path) -> None:
    """Histograms are counts, so subtracting them bucket-wise is what makes this work."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    for _ in range(5):
        with profiler.phase("step"):
            time.sleep(0.001)
    profiler.deltas()
    for _ in range(5):
        with profiler.phase("step"):
            time.sleep(0.020)
    interval = profiler.deltas()[("step",)]
    profiler.close()

    assert interval.hist.count == 5
    assert interval.hist.quantile(0.5) == pytest.approx(0.020e9, rel=0.6), (
        "the slow interval's median must not be dragged down by the fast one before it"
    )


def test_the_first_delta_read_does_not_alias_the_live_tree(tmp_path: Path) -> None:
    """The first call has no baseline and returns the tree itself unless it is copied."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("step"):
        pass
    first = profiler.deltas()
    with profiler.phase("step"):
        pass
    profiler.close()

    assert first[("step",)].calls == 1, "a later phase must not mutate an earlier reading"


def test_a_raising_export_callback_does_not_stop_the_flushes(tmp_path: Path) -> None:
    """An exporter losing its connection must not cost the run its remaining flushes."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=0.05, sample_interval_s=None,
    )
    seen: list[int] = []

    def broken(_tree: PhaseTree) -> None:
        raise RuntimeError("exporter is down")

    def working(tree: PhaseTree) -> None:
        seen.append(len(tree))

    profiler.on_snapshot(broken)
    profiler.on_snapshot(working)
    with profiler.phase("train"):
        pass
    deadline = time.monotonic() + 5.0
    while len(seen) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    profiler.close()

    assert len(seen) >= 3, "a raising callback stopped the ones after it, or the timer"
    assert profiler._callback_failures >= 3  # noqa: SLF001


def test_deltas_inside_a_snapshot_callback_see_the_interval(tmp_path: Path) -> None:
    """The documented combination for live export."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=0.05, sample_interval_s=None,
    )
    intervals: list[int] = []

    def export(_tree: PhaseTree) -> None:
        step = profiler.deltas().get(("train",))
        intervals.append(step.calls if step else 0)

    profiler.on_snapshot(export)
    for _ in range(3):
        with profiler.phase("train"):
            pass
        time.sleep(0.06)
    profiler.close()

    assert sum(intervals) <= 3, "deltas must not re-report work already exported"
    assert any(intervals), "and must report the work that happened"


def test_on_snapshot_is_usable_as_a_decorator(tmp_path: Path) -> None:
    """The README documents the decorator form; returning None would bind the name to None."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    seen: list[int] = []

    @profiler.on_snapshot
    def export(tree: PhaseTree) -> None:
        seen.append(len(tree))

    profiler.close()

    assert callable(export), "the decorated function must survive as a function"
    export({})
    assert seen == [0]


# ── per-thread attribution ──────────────────────────────────────────────────


def _two_threads(profiler: Profiler) -> None:
    """A learner thread and a collector thread doing unrelated work in one process."""
    def learner() -> None:
        threading.current_thread().name = "learner"
        for _ in range(3):
            with profiler.phase("train_step"):
                time.sleep(0.005)

    def collector() -> None:
        threading.current_thread().name = "collector"
        for _ in range(3):
            with profiler.phase("drain_queue"):
                time.sleep(0.005)

    threads = [threading.Thread(target=learner), threading.Thread(target=collector)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_threads_are_merged_together_by_default(tmp_path: Path) -> None:
    """The existing shape: one process, one tree, whatever thread recorded it."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    _two_threads(profiler)
    tree = profiler.merged_tree()
    profiler.close()

    assert ("train_step",) in tree
    assert ("drain_queue",) in tree


def test_thread_names_separate_two_threads_of_one_process(tmp_path: Path) -> None:
    """``role`` is per process, so a learner and a collector in one process were both
    reported as "learner" and their very different waits averaged together."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, thread_names=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    _two_threads(profiler)
    tree = profiler.merged_tree()
    profiler.close()

    assert ("learner", "train_step") in tree
    assert ("collector", "drain_queue") in tree
    assert ("learner", "drain_queue") not in tree


def test_a_thread_node_carries_the_time_of_the_phases_beneath_it(tmp_path: Path) -> None:
    """A thread's root is never entered, so without this it is a row of zeros and the report
    suppresses the whole block."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, thread_names=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    _two_threads(profiler)
    tree = profiler.merged_tree()
    profiler.close()

    assert tree[("learner",)].wall_ns == tree[("learner", "train_step")].wall_ns
    assert tree[("learner",)].self_ns == 0, "the synthesised node claims no time of its own"


def test_thread_names_survive_the_snapshot_and_render(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, thread_names=True, role="learner",
        snapshot_interval_s=None, sample_interval_s=None,
    )
    _two_threads(profiler)
    profiler.close()

    run = merge_run(tmp_path)
    text = render(run)

    assert ("learner", "train_step") in run.tree
    assert "collector" in text
