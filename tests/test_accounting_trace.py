"""The trace timeline: the ring buffer, the clock, the arrows and the critical path.

The failures worth guarding here are the ones that produce a *plausible* picture rather than
an obviously broken one: a truncated trace that looks complete, a span attributed to the
wrong phase, an arrow pointing at the wrong producer, or a lane that reads as busy while it
sat blocked.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from lineprofiler.accounting import Profiler
from lineprofiler.accounting.profiler import _lifecycle_admits
from lineprofiler.accounting.snapshot import merge_run, new_run_id, read_trace
from lineprofiler.accounting.trace import (
    FLAG_AUTO,
    FLAG_SAMPLED,
    UNMEASURED,
    ClockAnchor,
    Link,
    Span,
    TraceBuffer,
    WorkerTrace,
)
from lineprofiler.accounting.tracealign import (
    AlignedTrace,
    Arrow,
    PlacedSpan,
    align_run,
    alignment_accuracy_note,
    concurrent_activity,
    critical_path,
    lane_busy_share,
    lane_working_share,
    lifecycle_segments,
    match_links,
    max_depth_of,
    overlap_ns,
    place_spans,
    to_common_epoch,
)

# ── the ring buffer ─────────────────────────────────────────────────────────


def test_the_buffer_keeps_the_newest_spans_and_counts_what_it_dropped() -> None:
    """A wrapped ring must report its loss: a truncated trace must never look complete."""
    buffer = TraceBuffer(capacity=4)
    phase = buffer.intern(("work",))
    for index in range(6):
        buffer.record(phase, 0, index * 10, index * 10 + 5, 1)

    spans, _ = buffer.drain()
    assert [span.t0_ns for span in spans] == [20, 30, 40, 50]
    assert buffer.dropped == 2


def test_a_buffer_filled_exactly_to_capacity_drops_nothing() -> None:
    """The off-by-one that would report a phantom loss on a perfectly-sized buffer."""
    buffer = TraceBuffer(capacity=4)
    phase = buffer.intern(("work",))
    for index in range(4):
        buffer.record(phase, 0, index, index + 1, 0)

    spans, _ = buffer.drain()
    assert len(spans) == 4
    assert buffer.dropped == 0


def test_draining_twice_does_not_repeat_spans() -> None:
    """Each span reaches the sidecar exactly once, or the timeline doubles its own work."""
    buffer = TraceBuffer(capacity=8)
    buffer.record(buffer.intern(("a",)), 0, 1, 2, 0)

    assert len(buffer.drain()[0]) == 1
    assert buffer.drain()[0] == []


def test_interning_is_stable_and_resolves_back_to_the_path() -> None:
    buffer = TraceBuffer(capacity=8)
    first = buffer.intern(("iteration", "mcts"))

    assert buffer.intern(("iteration", "mcts")) == first
    assert buffer.paths()[first] == ("iteration", "mcts")


def test_an_unmeasured_cpu_time_is_not_a_zero_one() -> None:
    """The sentinel exists so "we did not measure" cannot be drawn as "it never waited"."""
    buffer = TraceBuffer(capacity=4)
    phase = buffer.intern(("auto",))
    buffer.record(phase, 0, 0, 100, UNMEASURED)
    buffer.record(phase, 0, 0, 100, 0)
    spans, _ = buffer.drain()

    unmeasured, fully_blocked = spans
    assert not unmeasured.cpu_measured
    assert fully_blocked.cpu_measured
    assert fully_blocked.wait_ns == 100


def test_clearing_drops_the_interning_table_too() -> None:
    """After a fork the child's ids must not keep the parent's meanings."""
    buffer = TraceBuffer(capacity=4)
    buffer.intern(("parent",))
    buffer.record(0, 0, 1, 2, 0)
    buffer.clear()

    assert buffer.is_empty()
    assert buffer.paths() == []
    assert buffer.intern(("child",)) == 0


def test_a_zero_capacity_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TraceBuffer(capacity=0)


# ── the clock ───────────────────────────────────────────────────────────────


def test_one_anchor_maps_through_a_constant_offset() -> None:
    anchor = ClockAnchor(perf_ns=1_000, real_ns=1_700_000_000_000)

    assert to_common_epoch(1_500, [anchor]) == anchor.real_ns + 500


def test_two_anchors_correct_for_drift_between_the_clocks() -> None:
    """A long run's lanes slide apart without this; the correction is the point of anchors."""
    start = ClockAnchor(perf_ns=1_000, real_ns=1_700_000_000_000)
    # The wall clock advanced 1100 ns while the monotonic clock advanced 1000.
    end = ClockAnchor(perf_ns=2_000, real_ns=1_700_000_001_100)

    assert to_common_epoch(1_500, [start, end]) == start.real_ns + 550


def test_without_anchors_the_reading_is_returned_unchanged() -> None:
    """No anchor means nothing to map through; inventing an offset would be worse."""
    assert to_common_epoch(42, []) == 42


def test_the_accuracy_note_distinguishes_one_host_from_several() -> None:
    """Cross-host alignment is NTP-bounded, and the page has to say so."""
    assert "exact" in alignment_accuracy_note({"node0"})
    assert "clock synchronisation" in alignment_accuracy_note({"node0", "node1"})


# ── links and arrows ────────────────────────────────────────────────────────

_ANCHORS = [ClockAnchor(perf_ns=0, real_ns=0)]


def _link(channel: str, key: str, kind: str, at: int) -> Link:
    return Link(channel=channel, key=key, kind=kind, t_ns=at, thread_id=0)


def test_a_wait_matches_the_signal_that_released_it() -> None:
    arrows, unmatched = match_links({
        "actor": ([_link("batch", "1", "signal", 100)], _ANCHORS),
        "learner": ([_link("batch", "1", "wait", 150)], _ANCHORS),
    })

    assert not unmatched
    assert [(a.src_worker, a.dst_worker, a.delay_ns) for a in arrows] == [
        ("actor", "learner", 50),
    ]


def test_a_channel_reused_each_iteration_links_to_its_own_producer() -> None:
    """Keys keep iteration n's wait off iteration n-1's signal."""
    arrows, _ = match_links({
        "actor": ([_link("b", "1", "signal", 10), _link("b", "2", "signal", 200)], _ANCHORS),
        "learner": ([_link("b", "1", "wait", 50), _link("b", "2", "wait", 250)], _ANCHORS),
    })

    assert sorted(arrow.delay_ns for arrow in arrows) == [40, 50]


def test_an_unmatched_wait_is_reported_rather_than_raised() -> None:
    """Half-instrumented code must cost arrows, never a crash."""
    arrows, unmatched = match_links({"learner": ([_link("ghost", "9", "wait", 10)], _ANCHORS)})

    assert not arrows
    assert unmatched == [("learner", "ghost", "9")]


def test_a_signal_nobody_waited_on_is_not_an_error() -> None:
    arrows, unmatched = match_links({"actor": ([_link("b", "1", "signal", 10)], _ANCHORS)})

    assert not arrows
    assert not unmatched


# ── derived views ───────────────────────────────────────────────────────────


def _span(worker: str, path: tuple[str, ...], t0: int, t1: int, cpu: int = -1) -> PlacedSpan:
    return PlacedSpan(
        worker=worker, role="r", thread_id=0, path=path,
        t0_ns=t0, t1_ns=t1, cpu_ns=cpu, flags=0,
    )


def test_a_blocked_lane_reads_as_occupied_but_not_as_working() -> None:
    """The gap between the two columns *is* the waiting, and is the page's whole point."""
    trace = AlignedTrace(spans=[
        _span("learner", ("queue_get",), 0, 100, cpu=0),
        _span("learner", ("train",), 100, 150, cpu=50),
    ])

    assert lane_busy_share(trace, "learner#0") == pytest.approx(100.0)
    assert lane_working_share(trace, "learner#0") == pytest.approx(100 * 50 / 150)


def test_nested_spans_do_not_double_count_cpu_time() -> None:
    """A parent's cpu_ns already includes its children's."""
    trace = AlignedTrace(spans=[
        _span("w", ("parent",), 0, 100, cpu=100),
        _span("w", ("parent", "child"), 10, 20, cpu=10),
    ])

    assert lane_working_share(trace, "w#0") == pytest.approx(10.0)


def test_a_lane_with_no_measured_cpu_reports_unknown_not_idle() -> None:
    trace = AlignedTrace(spans=[_span("w", ("auto",), 0, 100)])

    assert lane_working_share(trace, "w#0") == -1.0
    assert trace.spans[0].wait_pct == -1.0


def test_the_critical_path_crosses_to_the_worker_that_caused_the_wait() -> None:
    """The chain must leave the blocked lane and name the producer, or it explains nothing."""
    spans = [
        _span("actor0", ("gen",), 0, 100, cpu=100),
        _span("learner", ("queue_get",), 0, 100, cpu=0),
        _span("learner", ("train",), 100, 150, cpu=50),
    ]
    trace = AlignedTrace(
        spans=spans,
        arrows=[Arrow("b", "1", "actor0", "learner", 100, 100)],
    )

    chain = critical_path(trace)

    assert ("actor0", "gen") in [(span.worker, span.name) for span in chain]


def test_the_critical_path_follows_a_wait_recorded_just_after_the_span() -> None:
    """``wait_on`` is documented as running *after* the blocking get returns, so its
    timestamp lands microseconds past the span it belongs to.

    Requiring strict containment matched nothing on a real pipeline, and the chain never left
    the blocked worker's lane — it said *that* the learner waited without ever naming who it
    waited for, which is the one thing it exists to do.
    """
    spans = [
        _span("actor0", ("gen",), 0, 100, cpu=100),
        _span("learner", ("queue_get",), 0, 100, cpu=0),
        _span("learner", ("train",), 120, 170, cpu=50),
    ]
    trace = AlignedTrace(
        spans=spans,
        # The wait is stamped 10 ns after queue_get ended, as the real idiom produces.
        arrows=[Arrow("b", "1", "actor0", "learner", 100, 110)],
    )

    chain = critical_path(trace)

    assert "actor0" in {span.worker for span in chain}


def test_the_critical_path_of_an_empty_trace_is_empty() -> None:
    assert critical_path(AlignedTrace()) == []


# ── recording, end to end ───────────────────────────────────────────────────


def test_named_phases_become_spans_with_their_full_path(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("iteration"), profiler.phase("work"):
        pass

    run = merge_run(tmp_path, with_trace=True)
    trace = run.workers[0].trace
    recorded = {trace.path_of(span.phase_id) for span in trace.spans}

    assert ("iteration",) in recorded
    assert ("iteration", "work") in recorded


def test_tracing_is_off_unless_asked_for(tmp_path: Path) -> None:
    """The default must cost nothing: no buffer, no sidecar, no spans."""
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler, profiler.phase("iteration"):
        pass

    run = merge_run(tmp_path, with_trace=True)

    assert run.workers[0].trace.spans == []
    assert not list(tmp_path.rglob("*.trace"))


def test_the_environment_variable_turns_tracing_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 0 with zero code changes: a launcher exports one variable."""
    monkeypatch.setenv("LINEPROFILER_TRACE", "1")
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler, profiler.phase("iteration"):
        pass

    assert merge_run(tmp_path, with_trace=True).workers[0].trace.spans


def test_an_unrecognised_trace_mode_is_rejected(tmp_path: Path) -> None:
    """A typo in a launcher script must not silently disable the timeline."""
    with pytest.raises(ValueError, match="trace must be"):
        Profiler(run_dir=tmp_path, enabled=True, trace="yes-please")


def test_signal_and_wait_on_are_no_ops_without_tracing(tmp_path: Path) -> None:
    """The calls stay safe in code that is only sometimes profiled."""
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler:
        profiler.signal("batch", 1)
        profiler.wait_on("batch", 1)

    assert merge_run(tmp_path, with_trace=True).workers[0].trace.links == []


def test_the_module_level_helpers_no_op_with_nothing_installed() -> None:
    from lineprofiler.accounting import signal_ready, wait_on

    signal_ready("batch", 1)
    wait_on("batch", 1)


def test_a_sampled_phase_records_only_the_entries_it_measured(tmp_path: Path) -> None:
    """A timeline must not invent the entries sampling skipped."""
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler:
        for _ in range(10):
            with profiler.phase("step", sample=0.5):
                pass

    trace = merge_run(tmp_path, with_trace=True).workers[0].trace
    steps = [s for s in trace.spans if trace.path_of(s.phase_id) == ("step",)]

    assert 0 < len(steps) < 10
    assert all(span.flags & FLAG_SAMPLED for span in steps)


def test_a_torn_final_line_costs_only_that_batch(tmp_path: Path) -> None:
    """Append-only means a killed worker loses its last write and nothing before it."""
    sidecar = tmp_path / "w.trace"
    good = (
        '{"paths": ["a"], "anchors": [], "dropped": 0, "dropped_links": 0, '
        '"spans": [[0, 0, 0, 10, 5, 0]], "links": []}'
    )
    sidecar.write_text(good + "\n" + '{"paths": ["a"], "spans": [[0, 0', encoding="utf-8")

    trace = read_trace(sidecar)

    assert len(trace.spans) == 1


def test_a_missing_sidecar_reads_as_an_empty_trace(tmp_path: Path) -> None:
    assert read_trace(tmp_path / "absent.trace").spans == []


def test_a_span_with_an_unknown_phase_id_is_shown_not_dropped() -> None:
    """Dropping it would silently shorten the very lane being explained."""
    assert WorkerTrace().path_of(7) == ("(unknown)",)


def test_sequential_threads_get_their_own_lanes(tmp_path: Path) -> None:
    """CPython reuses ``threading.get_ident()`` once a thread has exited.

    Keying lanes on it merged three threads that ran one after another onto a single lane,
    each apparently doing all three threads' work — a confident wrong picture, not a missing
    one. The id comes from the per-thread state object instead, which is never recycled.
    """
    import threading

    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )

    def work(index: int) -> None:
        with profiler.phase(f"t{index}"):
            pass

    with profiler:
        for index in range(4):
            thread = threading.Thread(target=work, args=(index,))
            thread.start()
            thread.join()  # each exits before the next starts, so idents are reused

    trace = merge_run(tmp_path, with_trace=True).workers[0].trace
    names_by_lane: dict[int, set[str]] = {}
    for span in trace.spans:
        names_by_lane.setdefault(span.thread_id, set()).add(trace.path_of(span.phase_id)[-1])

    assert len(names_by_lane) == 4
    assert all(len(names) == 1 for names in names_by_lane.values())


def test_concurrent_threads_get_distinct_lanes(tmp_path: Path) -> None:
    """Two threads assigning an id at once must not be handed the same one."""
    import threading

    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )

    def work(index: int) -> None:
        for _ in range(100):
            with profiler.phase(f"t{index}"):
                pass

    with profiler:
        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    trace = merge_run(tmp_path, with_trace=True).workers[0].trace

    assert len({span.thread_id for span in trace.spans}) == 4


# ── multiprocess ────────────────────────────────────────────────────────────


def _traced_worker(run_dir: str, run_id: str, role: str) -> None:
    profiler = Profiler(
        run_dir=run_dir, role=role, run_id=run_id, enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler:
        with profiler.phase("work"):
            time.sleep(0.01)
        profiler.signal("batch", role)


@pytest.mark.parametrize("method", ["spawn", "fork", "forkserver"])
def test_every_worker_contributes_its_own_lane(tmp_path: Path, method: str) -> None:
    """Two processes must not collapse onto one lane, however the launcher named them."""
    context: mp.context.DefaultContext = mp.get_context(method)  # type: ignore[assignment]
    run_id = new_run_id()
    processes = [
        context.Process(target=_traced_worker, args=(str(tmp_path), run_id, role))
        for role in ("actor", "learner")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)

    aligned = align_run(merge_run(tmp_path, with_trace=True))

    assert len(aligned.lanes) == 2
    assert len({span.worker for span in aligned.spans}) == 2


def test_a_forked_child_does_not_inherit_the_parents_spans(tmp_path: Path) -> None:
    """An inherited buffer would report the parent's work as the child's."""
    import os

    profiler = Profiler(
        run_dir=tmp_path, role="parent", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler:
        with profiler.phase("parent_only"):
            pass
        child = os.fork()
        if child == 0:  # pragma: no cover - the child exits before pytest sees it
            with profiler.phase("child_only"):
                pass
            profiler.close()
            os._exit(0)
        os.waitpid(child, 0)

    run = merge_run(tmp_path, with_trace=True)
    per_worker = {
        worker.pid: {worker.trace.path_of(s.phase_id)[0] for s in worker.trace.spans}
        for worker in run.workers
    }
    children = [names for pid, names in per_worker.items() if pid != os.getpid()]

    assert children, "the forked child wrote no trace"
    assert all("parent_only" not in names for names in children)


# ── nesting depth ───────────────────────────────────────────────────────────


def _depths_of(run_dir: Path) -> dict[str, int]:
    """Deepest drawn level per phase path, from a real recording rather than a fixture."""
    aligned = align_run(merge_run(run_dir, with_trace=True))
    return {"/".join(span.path): span.depth for span in aligned.spans}


def test_a_named_phase_takes_its_depth_from_its_path(tmp_path: Path) -> None:
    """The phase path *is* the call stack, so depth costs nothing to know and must be exact."""
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("outer"), profiler.phase("middle"), profiler.phase("inner"):
        pass

    depths = _depths_of(tmp_path)
    assert depths["outer"] == 0
    assert depths["outer/middle"] == 1
    assert depths["outer/middle/inner"] == 2


def test_max_depth_reports_the_deepest_call_on_a_lane() -> None:
    """What the renderer sizes a lane from: too small a value would clip the deepest row."""
    trace = AlignedTrace(spans=[
        PlacedSpan("w", "r", 0, ("a",), 0, 100, -1, 0, depth=0),
        PlacedSpan("w", "r", 0, ("a", "b"), 10, 90, -1, 0, depth=1),
        PlacedSpan("w", "r", 0, ("a", "b", "c"), 20, 80, -1, 0, depth=2),
    ])

    assert max_depth_of(trace, "w#0") == 2
    assert max_depth_of(trace, "absent#0") == 0


def test_an_auto_span_gets_its_depth_from_containment() -> None:
    """An auto-derived path is a qualname, not an ancestry.

    Without deriving depth by containment every auto span would sit at depth 0, and a whole
    auto-traced lane would collapse into the single overpainted row this layout exists to fix.
    """
    trace = WorkerTrace(paths=[("caller",), ("callee",), ("deeper",)])
    trace.spans.extend([
        Span(phase_id=0, thread_id=0, t0_ns=0, t1_ns=1000, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
        Span(phase_id=1, thread_id=0, t0_ns=100, t1_ns=900, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
        Span(phase_id=2, thread_id=0, t0_ns=200, t1_ns=300, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
    ])

    placed = place_spans(trace, "w", "r")

    assert [span.depth for span in placed] == [0, 1, 2]


def test_sequential_auto_calls_stay_on_one_row() -> None:
    """Containment, not arrival order: two calls that merely follow each other do not nest."""
    trace = WorkerTrace(paths=[("first",), ("second",)])
    trace.spans.extend([
        Span(phase_id=0, thread_id=0, t0_ns=0, t1_ns=100, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
        Span(phase_id=1, thread_id=0, t0_ns=100, t1_ns=200, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
    ])

    assert [span.depth for span in place_spans(trace, "w", "r")] == [0, 0]


def test_one_threads_nesting_never_deepens_another() -> None:
    """Depth is per lane. Threads interleave in time, and lanes are drawn separately."""
    trace = WorkerTrace(paths=[("a",), ("b",)])
    trace.spans.extend([
        Span(phase_id=0, thread_id=0, t0_ns=0, t1_ns=1000, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
        Span(phase_id=1, thread_id=1, t0_ns=100, t1_ns=900, cpu_ns=UNMEASURED, flags=FLAG_AUTO),
    ])

    assert [span.depth for span in place_spans(trace, "w", "r")] == [0, 0]


# ── overlap and concurrency ─────────────────────────────────────────────────


def _role_span(role: str, t0: int, t1: int, cpu: int = -1) -> PlacedSpan:
    """A span carrying a real role, which the shared ``_span`` helper does not."""
    return PlacedSpan(
        worker=role, role=role, thread_id=0, path=("p",),
        t0_ns=t0, t1_ns=t1, cpu_ns=cpu, flags=0,
    )


def test_disjoint_intervals_do_not_overlap() -> None:
    assert overlap_ns([(0, 100)], [(200, 300)]) == 0


def test_identical_intervals_overlap_entirely() -> None:
    assert overlap_ns([(0, 100)], [(0, 100)]) == 100


def test_nested_intervals_are_counted_once() -> None:
    """A phase inside a phase must not let one lane count the same nanosecond twice."""
    assert overlap_ns([(0, 100), (10, 50), (20, 30)], [(0, 100)]) == 100


def test_overlap_is_symmetric() -> None:
    assert overlap_ns([(0, 100)], [(60, 200)]) == overlap_ns([(60, 200)], [(0, 100)])


def test_a_blocked_worker_reports_the_share_another_was_busy() -> None:
    """The acceptance test: A blocks 100 ms, B is busy exactly 60 of them.

    This is the question behind every 96%-wait phase — a hang and a queue look identical in
    the phase table and have nothing in common as problems.
    """
    blocked = _role_span("actor", 0, 100, cpu=0)
    trace = AlignedTrace(spans=[blocked, _role_span("server", 20, 80)])

    shares = concurrent_activity(trace, [blocked])

    assert shares["server"] == pytest.approx(60.0)


def test_a_role_spread_over_lanes_is_counted_once_not_once_per_lane() -> None:
    """Two threads of one server covering the same window is 100% busy, not 200%."""
    blocked = _role_span("actor", 0, 100, cpu=0)
    first = PlacedSpan(
        worker="server", role="server", thread_id=0, path=("a",),
        t0_ns=0, t1_ns=100, cpu_ns=-1, flags=0,
    )
    second = PlacedSpan(
        worker="server", role="server", thread_id=1, path=("b",),
        t0_ns=0, t1_ns=100, cpu_ns=-1, flags=0,
    )
    trace = AlignedTrace(spans=[blocked, first, second])

    assert concurrent_activity(trace, [blocked])["server"] == pytest.approx(100.0)


def test_a_stall_reports_no_concurrent_activity() -> None:
    """Nothing running anywhere is the other answer, and must be distinguishable."""
    blocked = _role_span("actor", 0, 100, cpu=0)
    trace = AlignedTrace(spans=[blocked])

    assert concurrent_activity(trace, [blocked]) == {}


# ── request lifecycles ──────────────────────────────────────────────────────


def _mark(key: str, kind: str, at: int) -> Link:
    return Link(channel="inference", key=key, kind=kind, t_ns=at, thread_id=0)


def test_a_lifecycle_decomposes_into_named_segments() -> None:
    """The acceptance test: a known 50 ms before admitting and 20 ms computing.

    One ``queue_wait`` bar fuses intervals whose remedies point in opposite directions —
    batch harder, shrink the window, cheaper model, fewer hops. This is what separates them.
    """
    trace = AlignedTrace(lifecycle_marks=[
        _mark("7", "begin", 0),
        _mark("7", "mark:admitted", 50_000_000),
        _mark("7", "mark:computed", 70_000_000),
        _mark("7", "end", 70_500_000),
    ])

    segments = {segment.name: segment for segment in lifecycle_segments(trace)["inference"]}

    assert segments["begin → admitted"].total_ns == 50_000_000
    assert segments["admitted → computed"].total_ns == 20_000_000
    assert segments["computed → end"].total_ns == 500_000


def test_segments_are_totalled_across_requests() -> None:
    """One request's breakdown answers the question only by accident."""
    marks = []
    for index in range(3):
        base = index * 1_000_000_000
        marks.extend([
            _mark(str(index), "begin", base),
            _mark(str(index), "mark:admitted", base + 10_000_000),
            _mark(str(index), "end", base + 15_000_000),
        ])

    channels = lifecycle_segments(AlignedTrace(lifecycle_marks=marks))
    segments = {segment.name: segment for segment in channels["inference"]}

    assert segments["begin → admitted"].total_ns == 30_000_000
    assert segments["begin → admitted"].count == 3
    assert segments["begin → admitted"].mean_ns == pytest.approx(10_000_000)


def test_an_unfinished_lifecycle_contributes_nothing() -> None:
    """Closing it at the last mark would turn an unfinished request into a fast one."""
    trace = AlignedTrace(lifecycle_marks=[
        _mark("7", "begin", 0),
        _mark("7", "mark:admitted", 50_000_000),
    ])

    assert lifecycle_segments(trace) == {}


def test_checkpoints_out_of_order_are_dropped_not_counted_as_negative() -> None:
    """Cross-host skew is ordinary; a segment that cannot have happened is not."""
    trace = AlignedTrace(lifecycle_marks=[
        _mark("7", "mark:admitted", 0),
        _mark("7", "begin", 50_000_000),
        _mark("7", "end", 70_000_000),
    ])

    assert lifecycle_segments(trace) == {}


def test_lifecycle_marks_do_not_become_arrows() -> None:
    """The two mechanisms answer different questions and must not contaminate each other."""
    links = [
        _link("inference", "7", "begin", 0),
        _link("inference", "7", "end", 10),
        _link("inference", "7", "signal", 5),
        _link("inference", "7", "wait", 8),
    ]

    arrows, unmatched = match_links({"w": (links, _ANCHORS)})

    assert len(arrows) == 1
    assert unmatched == []


def test_a_sampled_lifecycle_keeps_or_drops_every_mark_of_one_request(tmp_path: Path) -> None:
    """Selection is by key, not a counter: the marks are recorded in different processes.

    A counter would admit a request's ``admitted`` mark on the server while dropping its
    ``begin`` on the client, leaving segments no request ever experienced.
    """
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler:
        for index in range(200):
            profiler.trace_mark("inference", index, "admitted", sample=0.1)
        profiler.snapshot()

    recorded = merge_run(tmp_path, with_trace=True).workers[0].trace.links
    kept = {link.key for link in recorded}

    assert 5 < len(kept) < 60, f"expected roughly a tenth of 200, kept {len(kept)}"
    # Every mark of a kept key survives, which is what a second pass must reproduce.
    for key in list(kept)[:5]:
        assert _lifecycle_admits(int(key), 0.1)
