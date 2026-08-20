"""Tests for the ranked findings and the phase summary.

These are the page's *conclusions*, so the assertions are about whether a conclusion is
correct and whether a wrong one is reachable — not about wording. A finding that names the
wrong phase, double-counts a wait, or calls a queue a stall is worse than no finding at all,
because the reader has no way to check it short of re-deriving the run by hand.

Most traces here are built directly rather than by running a profiler: a finding about "the
lane that was idle 80% of the run" needs a run that is idle exactly 80%, which a real workload
cannot promise. The end-to-end tests at the bottom are the exception — they are about the
wiring between the derivation and each renderer, so they run a real profiler, where a
hand-made trace would prove a renderer works on data no profiler produces.
"""
from __future__ import annotations

import time
from pathlib import Path

from lineprofiler.accounting import Profiler
from lineprofiler.accounting.cli import main
from lineprofiler.accounting.findings import phase_totals, rank_findings
from lineprofiler.accounting.report import render, report_as_dict
from lineprofiler.accounting.snapshot import merge_run
from lineprofiler.accounting.tracealign import AlignedTrace, Arrow, PlacedSpan

_MS = 1_000_000


def _blocked_run(run_dir: Path) -> None:
    """A real profiled run whose learner genuinely blocks, for the end-to-end paths.

    Built by running a profiler rather than by hand: these tests are about the wiring between
    the derivation and each renderer, so a hand-made trace would prove the renderer works on
    data no profiler produces.
    """
    profiler = Profiler(
        run_dir=run_dir, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
        trace=True, measure_cpu=True,
    )
    with profiler:
        for _ in range(3):
            with profiler.phase("iteration"), profiler.phase("queue_get"):
                time.sleep(0.02)


def _span(
    worker: str,
    path: tuple[str, ...],
    t0_ms: int,
    t1_ms: int,
    cpu_ms: int | None = None,
    role: str = "",
    depth: int = 0,
) -> PlacedSpan:
    """One span in milliseconds, with ``cpu_ms=None`` meaning CPU time was not measured."""
    return PlacedSpan(
        worker=worker,
        role=role or worker,
        thread_id=0,
        path=path,
        t0_ns=t0_ms * _MS,
        t1_ns=t1_ms * _MS,
        cpu_ns=-1 if cpu_ms is None else cpu_ms * _MS,
        flags=0,
        depth=depth,
    )


def _trace(spans: list[PlacedSpan], arrows: list[Arrow] | None = None) -> AlignedTrace:
    """An aligned trace over ``spans``, with lanes and roles derived from them."""
    lanes = list(dict.fromkeys(span.lane for span in spans))
    return AlignedTrace(
        spans=spans,
        arrows=arrows or [],
        lanes=lanes,
        roles={span.lane: span.role for span in spans},
    )


# --------------------------------------------------------------------------- #
# What the findings say
# --------------------------------------------------------------------------- #
def test_a_blocked_phase_is_named_with_what_it_cost() -> None:
    """The core finding: a phase that spent its time waiting, and its share of the run."""
    trace = _trace([
        _span("learner", ("iteration",), 0, 100, cpu_ms=100, role="learner"),
        _span("learner", ("wait",), 100, 900, cpu_ms=0, role="learner", depth=0),
    ])

    findings = rank_findings(trace)

    assert any("wait" in finding.headline for finding in findings)
    blocked = next(finding for finding in findings if "wait" in finding.headline)
    assert blocked.kind == "blocked-phase"
    assert blocked.anchor == "wait"


def test_a_parent_that_only_waits_in_its_child_is_not_reported_twice() -> None:
    """One stall must not appear as two findings, or it out-ranks everything real.

    ``iteration`` here does nothing but call ``queue_get``, so it inherits all of its wait.
    Reporting both states one fact twice and pushes the genuine second-place finding off the
    list entirely.
    """
    trace = _trace([
        _span("learner", ("iteration",), 0, 1000, cpu_ms=10, role="learner"),
        _span("learner", ("iteration", "queue_get"), 5, 995, cpu_ms=5,
              role="learner", depth=1),
    ])

    paths = [finding.anchor for finding in rank_findings(trace)]

    assert "iteration/queue_get" in paths
    assert "iteration" not in paths


def test_a_wait_released_by_a_signal_is_called_a_queue_not_a_stall() -> None:
    """A recorded signal/wait_on pair settles the question; concurrency only infers it.

    This is the regression that matters most: the producer works in short bursts across a long
    wait, so its concurrent-activity share is low and the inference alone would call this a
    stall — while an arrow explicitly names who released it.
    """
    spans = [
        _span("learner", ("queue_get",), 0, 1000, cpu_ms=0, role="learner"),
        _span("actor", ("publish",), 900, 950, cpu_ms=50, role="actor"),
    ]
    arrows = [
        Arrow(
            channel="batch", key="0",
            src_worker="actor", dst_worker="learner",
            src_t_ns=940 * _MS, dst_t_ns=1000 * _MS,
        ),
    ]

    finding = next(
        item for item in rank_findings(_trace(spans, arrows)) if item.anchor == "queue_get"
    )

    assert "queue" in finding.detail
    assert "actor" in finding.detail


def test_a_wait_with_nothing_else_running_is_called_a_stall() -> None:
    """The opposite verdict, so the queue label cannot be the only thing it ever says."""
    trace = _trace([
        _span("solo", ("blocked",), 0, 1000, cpu_ms=0, role="solo"),
    ])

    finding = next(item for item in rank_findings(trace) if item.anchor == "blocked")

    assert "stall" in finding.detail


def test_idle_lanes_of_one_role_collapse_into_a_single_finding() -> None:
    """Sixteen idle actors are one finding about the pipeline, not sixteen about actors."""
    spans = [_span("learner", ("train",), 0, 1000, cpu_ms=1000, role="learner")]
    for index in range(4):
        spans.append(_span(f"actor{index}", ("step",), 0, 50, cpu_ms=50, role="actor"))

    idle = [finding for finding in rank_findings(_trace(spans)) if finding.kind == "idle-lane"]

    assert len(idle) == 1
    assert len(idle[0].lanes) == 4


def test_findings_are_ranked_by_what_they_cost() -> None:
    """The list is a ranking; an unordered one would bury the answer under a detail."""
    trace = _trace([
        _span("w", ("small",), 0, 1000, cpu_ms=900, role="w"),
        _span("w", ("huge",), 1000, 9000, cpu_ms=0, role="w"),
    ])

    findings = rank_findings(trace)

    assert findings == sorted(findings, key=lambda finding: -finding.cost_pct)


def test_an_unmeasured_span_produces_no_blocked_finding() -> None:
    """Unknown wait is never reported as a wait of zero, nor as a wait of everything."""
    trace = _trace([_span("w", ("derived",), 0, 1000, cpu_ms=None, role="w")])

    assert not [f for f in rank_findings(trace) if f.kind == "blocked-phase"]


def test_a_healthy_run_produces_no_findings() -> None:
    """Inventing a finding to fill the section would make every page look broken."""
    spans = [
        _span("a", ("work",), 0, 1000, cpu_ms=1000, role="a"),
        _span("b", ("work",), 0, 1000, cpu_ms=1000, role="b"),
    ]

    assert rank_findings(_trace(spans)) == []


def test_an_empty_trace_is_not_an_error() -> None:
    assert rank_findings(AlignedTrace()) == []


# --------------------------------------------------------------------------- #
# The phase summary
# --------------------------------------------------------------------------- #
def test_phase_totals_rank_by_wall_time_across_lanes() -> None:
    """The summary answers 'what is this run made of', pooled over every lane."""
    spans = [
        _span("a", ("cheap",), 0, 10, cpu_ms=10, role="a"),
        _span("a", ("costly",), 10, 500, cpu_ms=500, role="a"),
        _span("b", ("costly",), 0, 400, cpu_ms=400, role="b"),
    ]

    totals = phase_totals(_trace(spans))

    assert totals[0].path == "costly"
    assert totals[0].calls == 2
    assert totals[0].lanes == 2


def test_self_time_excludes_nested_phases() -> None:
    """A wrapper must not out-rank the callee that actually spent the time."""
    spans = [
        _span("w", ("outer",), 0, 1000, cpu_ms=1000, role="w", depth=0),
        _span("w", ("outer", "inner"), 100, 900, cpu_ms=800, role="w", depth=1),
    ]

    totals = {total.path: total for total in phase_totals(_trace(spans))}

    assert totals["outer"].wall_ns == 1000 * _MS
    assert totals["outer"].self_ns == 200 * _MS
    assert totals["outer/inner"].self_ns == 800 * _MS


def test_an_unmeasured_phase_reports_its_wait_as_unknown() -> None:
    """``-1`` is the package's 'not measured'; zero would claim it never waited."""
    totals = phase_totals(_trace([_span("w", ("auto",), 0, 100, cpu_ms=None, role="w")]))

    assert totals[0].wait_pct == -1.0
    assert not totals[0].measured


# --------------------------------------------------------------------------- #
# Where findings surface
# --------------------------------------------------------------------------- #
def test_the_text_report_leads_with_the_same_findings(tmp_path: Path) -> None:
    """The terminal and the page must never disagree about what the bottleneck was."""
    _blocked_run(tmp_path)
    run = merge_run(tmp_path, with_trace=True)

    text = render(run)

    assert "FINDINGS" in text
    assert text.index("FINDINGS") < text.index("RESOURCES")


def test_a_run_without_a_trace_prints_no_findings_block(tmp_path: Path) -> None:
    """Findings come from spans; a phase tree alone cannot say who waited for whom.

    Inventing a weaker finding from totals would put a claim at the top of the report that
    nothing below it could support.
    """
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler, profiler.phase("work"):
        pass

    assert "FINDINGS" not in render(merge_run(tmp_path))


def test_the_json_document_carries_findings_for_a_gate(tmp_path: Path) -> None:
    """A CI gate needs the verdict as data, not the prose that explains it."""
    _blocked_run(tmp_path)

    document = report_as_dict(merge_run(tmp_path, with_trace=True))

    assert document["findings"]
    assert {"kind", "headline", "cost_pct", "anchor"} <= set(document["findings"][0])


def test_fail_over_exits_non_zero_only_above_the_threshold(tmp_path: Path) -> None:
    """The gate: a build fails when a finding crosses the line, and passes when it does not."""
    _blocked_run(tmp_path)
    page = tmp_path / "trace.html"

    over = main(["trace", str(tmp_path), "--fail-over", "1", "-o", str(page), "-q"])
    under = main(["trace", str(tmp_path), "--fail-over", "99.9", "-o", str(page), "-q"])

    assert over == 1
    assert under == 0


def test_the_gate_is_inert_until_a_threshold_is_set(tmp_path: Path) -> None:
    """Adding the timeline to a pipeline must not start failing builds on its own."""
    _blocked_run(tmp_path)
    page = tmp_path / "trace.html"

    assert main(["trace", str(tmp_path), "-o", str(page), "-q"]) == 0


def test_the_gate_still_writes_the_report_it_failed_on(tmp_path: Path) -> None:
    """A gate that fails without producing the evidence makes the failure undiagnosable."""
    _blocked_run(tmp_path)
    page = tmp_path / "trace.html"

    main(["trace", str(tmp_path), "--fail-over", "1", "-o", str(page), "-q"])

    assert page.read_text(encoding="utf-8").startswith("<!doctype html>")
