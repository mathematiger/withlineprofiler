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

from lineprofiler.accounting import DurationHistogram, PhaseStats, Profiler, merge_run
from lineprofiler.accounting.histogram import BUCKET_COUNT, bucket_index, bucket_lower_ns
from lineprofiler.accounting.phase import PhaseTree, merge_trees
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
