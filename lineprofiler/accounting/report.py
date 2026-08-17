"""Plain-text rendering of a merged run.

The report is organised by *role* — learner, actor, inference server, or whatever this
architecture calls its processes — because a single global percentage is misleading the
moment several workers run concurrently. Sixteen actors doing self-play will always
dominate a global pie chart, whether or not self-play is the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lineprofiler.accounting.analysis import (
    SampleAnalysis,
    analyse_processes,
    format_bytes,
    sparkline,
)
from lineprofiler.accounting.phasetree import PhasePath, PhaseStats, PhaseTree
from lineprofiler.accounting.provenance import source_of
from lineprofiler.accounting.snapshot import MergedRun, WorkerSnapshot, imbalance_of
from lineprofiler.accounting.tracealign import (
    AlignedTrace,
    PlacedSpan,
    align_run,
    concurrent_activity,
    lifecycle_segments,
    role_occupancy,
)

_WIDTH = 62
_RULE = "─" * _WIDTH

_CACHE_NOISE_FLOOR = 64 * 1024
"""Below one readahead window, cached reads are interpreter noise rather than your data."""

_STALE_AFTER_S = 300.0
"""How far a worker's last snapshot may trail the run's before it is called out. Wide enough
that an ordinary staggered shutdown stays quiet, tight enough to catch a worker whose flushes
died hours ago — the failure that used to be invisible."""


def render(run: MergedRun) -> str:
    """Return the full text report for a merged run.

    Test specifically:
        - golden-file comparison against a fixed synthetic run
        - a run with no phases and no samples renders without raising
        - unreadable workers are named in the output rather than silently dropped
        - a run with two roles renders a separate block for each
    """
    analysis = analyse_processes(run.samples_by_process())
    aligned = _aligned_or_none(run)
    occupancy = role_occupancy(aligned) if aligned is not None else {}
    blocks = [_header(run)]
    for role in run.roles:
        blocks.append(_role_block(run, role, occupancy.get(role), aligned))
    blocks.append(_lifecycle_block(aligned))
    blocks.append(_exact_io_block(run.tree))
    blocks.append(_io_block(analysis))
    blocks.append(_gpu_block(analysis))
    blocks.append(_memory_block(analysis))
    blocks.append(_backend_block(run))
    blocks.append(_losses_block(run))
    return "\n".join(block for block in blocks if block)


def report_as_dict(run: MergedRun) -> dict[str, Any]:
    """Return the same run as JSON-serialisable data, for asserting on rather than reading.

    A merged run is a machine-readable record of what actually executed — which roles
    started, which phases ran, how much work each did — so it makes a usable assertion target
    in a test or a CI gate. The text report answers that question for a human; this answers it
    for a script, without a caller having to re-derive the quantiles and shares itself.

    ``caveats`` is deliberately part of the document rather than a log line: a run that lost a
    worker, or merged a stale one, must not read as a complete result to a program either.

    Test specifically:
        - every phase in the text report appears here with the same numbers
        - a run read with ``with_samples=False`` omits ``resources`` rather than zeroing it
        - the document survives ``json.dumps`` with no custom encoder
    """
    document: dict[str, Any] = {
        "run": {
            "run_id": run.metadata.get("run_id"),
            "source": run.metadata.get("source", {}),
            "hosts": run.hosts,
            "processes": len(run.workers),
            "roles": run.roles,
            "runtime_s": max((w.written_at - w.started_at for w in run.workers), default=0.0),
            "imbalance": run.imbalance,
        },
        "roles": [
            {
                "role": role,
                "processes": len(run.workers_of(role)),
                "imbalance": imbalance_of(run.workers_of(role)),
                "phases": _phases_as_list(run.tree_of(role)),
            }
            for role in run.roles
        ],
        "workers": [
            {
                "pid": worker.pid,
                "role": worker.role,
                "host": worker.host,
                "rank": worker.rank,
                "started_at": worker.started_at,
                "written_at": worker.written_at,
                "wall_ns": worker.wall_ns,
                "write_failures": worker.write_failures,
            }
            for worker in run.workers
        ],
        "caveats": {
            "unreadable": [str(path) for path in run.unreadable],
            "superseded": [str(worker.path) for worker in run.superseded],
            "stale": [worker.label for worker in _stale_workers(run)],
        },
    }
    samples = run.samples_by_process()
    if samples:
        document["resources"] = _resources_as_dict(analyse_processes(samples))
    return document


def _phases_as_list(tree: PhaseTree) -> list[dict[str, Any]]:
    """Flatten a phase tree into rows, deepest-costing first, with derived quantiles."""
    return [
        {
            "phase": "/".join(path),
            "calls": stats.calls,
            "wall_ns": stats.wall_ns,
            "cpu_ns": stats.cpu_ns,
            "self_ns": stats.self_ns,
            # Pairs with wall_ns, never with self_ns: wait spans the whole phase, including
            # time inside children, so wait/self exceeds 100% for any parent that waits.
            "wait_ns": stats.wait_ns,
            "p50_ns": stats.hist.quantile(0.5),
            "p99_ns": stats.hist.quantile(0.99),
            "counters": dict(stats.counters),
        }
        for path, stats in sorted(tree.items(), key=lambda item: -item[1].self_ns)
        if path
    ]


def _resources_as_dict(analysis: SampleAnalysis) -> dict[str, Any]:
    """The sampled blocks: I/O at both counter layers, memory, and per-device GPU."""
    return {
        "io": {
            "read_bytes": analysis.totals.read_bytes,
            "write_bytes": analysis.totals.write_bytes,
            "read_chars": analysis.totals.read_chars,
            "write_chars": analysis.totals.write_chars,
            "cached_read_bytes": analysis.totals.cached_read_bytes,
            "unattributed_read_share": analysis.unattributed_read_share,
            "unattributed_write_share": analysis.unattributed_write_share,
            "gap_intervals": analysis.io_gap_intervals,
            "intervals": analysis.io_intervals,
        },
        "memory": {
            "peak_rss": analysis.memory.peak_rss,
            "last_rss": analysis.memory.last_rss,
            "growth_bytes": analysis.memory.growth_bytes,
            "slope_bytes_per_s": analysis.memory.slope_bytes_per_s,
        },
        "gpu": {
            "peak_cuda_alloc": analysis.peak_cuda_alloc,
            "peak_cuda_reserved": analysis.peak_cuda_reserved,
            "devices": [
                {"index": d.index, "busy_mean": d.busy_mean, "ours_mean": d.ours_mean}
                for d in analysis.gpu_devices
            ],
        },
    }


def format_ns(value: float) -> str:
    """Render a nanosecond duration with a unit that keeps three significant digits."""
    if value >= 3.6e12:
        hours, remainder = divmod(value, 3.6e12)
        return f"{int(hours)}h {int(remainder // 6e10):02d}m"
    if value >= 6e10:
        return f"{int(value // 6e10)}m {int((value % 6e10) // 1e9):02d}s"
    if value >= 1e9:
        return f"{value / 1e9:.2f}s"
    if value >= 1e6:
        return f"{value / 1e6:.1f}ms"
    if value >= 1e3:
        return f"{value / 1e3:.1f}us"
    return f"{value:.0f}ns"


def format_label(text: str, width: int) -> str:
    """Truncate a phase label from the left to ``width``, keeping the tail and marking the cut.

    The tail is the informative end of a phase path — the leaf is what a reader greps for —
    so an over-long path loses its head rather than its leaf. The ellipsis is what makes that
    safe: an unmarked left-truncation prints a name that does not exist. A plain ``[-26:]``
    turned ``train_step/forward_backward`` into ``rain_step/forward_backward``, and a plain
    ``[-27:]`` in the comparison table turned ``iteration/checkpoint_to_object_store`` into
    ``/checkpoint_to_object_store``.

    Callers pad the result into a field at least one column wider than ``width``. That is the
    second half of the fix and the ellipsis does not replace it: at exactly the column width
    the old label also swallowed the gap and ran into the heading beside it, printing
    ``…forward_backwardr        0 B``.
    """
    if len(text) <= width:
        return text
    return "…" + text[-(width - 1):]


def _header(run: MergedRun) -> str:
    """Runtime, process and worker counts, the nodes involved, and the run's identity."""
    runtime = max((w.written_at - w.started_at for w in run.workers), default=0.0)
    roles = ", ".join(f"{role} x{len(run.workers_of(role))}" for role in run.roles) or "none"
    lines = [
        f"Runtime {format_ns(runtime * 1e9)}   "
        # One worker file is one process. Counting distinct pids undercounted every
        # multi-node run: pid namespaces are per-node, so ranks on different nodes collide.
        f"Processes {len(run.workers)}   "
        f"Roles {roles}",
        _hosts_line(run),
    ]
    # Omitted rather than guessed at when absent: a run from a directory that is not a
    # repository, or one written before this was recorded, must not be given a revision it
    # never claimed.
    source = source_of(run.metadata)
    if source:
        lines.append(source)
    return "\n".join(lines)


def _hosts_line(run: MergedRun) -> str:
    """Name the nodes involved, and the attempt, so a report cannot be mistaken for another."""
    hosts = run.hosts
    run_id = run.metadata.get("run_id")
    suffix = f"   Run {run_id}" if run_id else ""
    if len(hosts) > 4:
        return f"Hosts {', '.join(hosts[:4])} +{len(hosts) - 4} more ({len(hosts)} nodes){suffix}"
    if len(hosts) > 1:
        return f"Hosts {', '.join(hosts)} ({len(hosts)} nodes){suffix}"
    single = hosts[0] if hosts else str(run.metadata.get("host", "?"))
    return f"Host {single}{suffix}"


def _aligned_or_none(run: MergedRun) -> AlignedTrace | None:
    """The aligned trace, or ``None`` when the run recorded none.

    Tracing is off by default, so most reports will not have one. Absent is absent — a figure
    derived from a timeline must not be shown for a run that has no timeline.
    """
    # Links as well as spans: a worker may record only lifecycle checkpoints — instrumenting
    # a queue boundary does not require naming a phase around it — and gating on spans alone
    # dropped the whole request breakdown for exactly that caller.
    if not any(
        getattr(worker, "trace", None) and (worker.trace.spans or worker.trace.links)
        for worker in run.workers
    ):
        return None
    return align_run(run)


def _role_block(
    run: MergedRun,
    role: str,
    occupancy: tuple[float, float] | None = None,
    aligned: AlignedTrace | None = None,
) -> str:
    """Render one role's phase share and its heaviest phases."""
    tree = run.tree_of(role)
    workers = run.workers_of(role)
    total = sum(stats.wall_ns for path, stats in tree.items() if len(path) == 1)
    if total <= 0:
        return ""

    lines = [
        "",
        f"{role.upper()}  ({len(workers)} process{'es' if len(workers) != 1 else ''}, "
        f"imbalance {imbalance_of(workers):.2f})",
        # Name what the percentages are *of*. Four different denominators are plausible here
        # — wall clock, busy time, summed self time, the role's total across processes — and
        # they give materially different readings, so leaving the reader to infer it means
        # leaving them to infer it wrongly.
        f"  % of {_share_basis(len(workers))}",
        _RULE,
    ]
    lines.extend(_share_rows(tree, len(workers)))
    lines.extend(_occupancy_rows(occupancy))
    lines.append("")
    lines.extend(_dominant_rows(tree))
    lines.extend(_concurrency_rows(role, aligned))
    lines.extend(_iteration_rows(tree))
    return "\n".join(lines)


def _concurrency_rows(role: str, aligned: AlignedTrace | None, limit: int = 3) -> list[str]:
    """What everyone else was doing while this role's most wait-heavy phase was blocked.

    A hang and a queue look identical in a phase table and have nothing in common as problems.
    Naming the lanes that were busy during the wait separates them: work happening elsewhere
    means queueing, silence everywhere means a hang.
    """
    if aligned is None:
        return []
    blocked = _blocked_spans(aligned, role)
    if not blocked:
        return []
    shares = concurrent_activity(aligned, blocked)
    busiest = sorted(shares.items(), key=lambda item: -item[1])[:limit]
    busiest = [(lane, share) for lane, share in busiest if share >= 1.0]
    if not busiest:
        return [
            "",
            f"  while {role} waited, no other lane was active — this is a stall, not a queue.",
        ]
    named = ", ".join(f"{lane} {share:.0f}%" for lane, share in busiest)
    return ["", f"  while {role} waited, concurrently active: {named}"]


def _blocked_spans(aligned: AlignedTrace, role: str) -> list[PlacedSpan]:
    """The role's spans that spent most of their time off-CPU.

    Overlapping and nested spans are harmless here: :func:`concurrent_activity` merges its
    windows before measuring, so the same blocked microsecond cannot be counted twice however
    deeply the phases nest.
    """
    return [
        span for span in aligned.spans
        if span.role == role and span.cpu_measured and span.duration_ns > 0
        and span.wait_ns > span.duration_ns * 0.5
    ]


def _occupancy_rows(occupancy: tuple[float, float] | None) -> list[str]:
    """State how much of the run this role held a phase open, and how much it was on a CPU.

    The gap between the two is the bottleneck statement — a role busy 97% and working 35% is
    waiting for something two thirds of the time, and neither figure says that alone. Both
    terms are defined inline rather than assumed: "busy" and "working" are ordinary words
    doing precise work here, and a reader who guesses at them guesses wrong.
    """
    if occupancy is None:
        return []
    busy, working = occupancy
    rows = ["", f"{'busy (phase open)':<28}{busy:>7.1f}%"]
    if working < 0:
        rows.append(f"{'working (on CPU)':<28}{'n/a':>8}")
        return rows
    rows.append(f"{'working (on CPU)':<28}{working:>7.1f}%")
    rows.append("  busy = a phase was open; working = on a CPU inside one. The gap is waiting.")
    return rows


def _share_basis(processes: int) -> str:
    """Name the denominator the share column divides by, and say when it is a sum.

    A total that exceeds the run's wall clock is correct for a multi-process role and reads
    as an error unless the summing is stated.
    """
    basis = "phase wall time at the first branching level"
    if processes > 1:
        return f"{basis}, summed over {processes} processes"
    return basis


@dataclass(frozen=True, slots=True)
class SiblingShare:
    """One row of the pipeline breakdown: a phase's wall time against its siblings' total.

    Carries the numerator and denominator rather than a precomputed percentage so every
    renderer divides them the same way. Two implementations of "share of parent" drifting
    apart is the wrong-numbers failure this layer is built to avoid.
    """

    name: str
    wall_ns: int
    total_ns: int

    @property
    def percent(self) -> float:
        """The share as a percentage, or 0.0 when nothing was measured."""
        return 100.0 * self.wall_ns / self.total_ns if self.total_ns else 0.0


def sibling_shares(tree: PhaseTree) -> list[SiblingShare]:
    """The pipeline breakdown: sibling phases as a share of their parent's wall time.

    A loop wrapped in a single outer phase would otherwise render as one 100% row, which
    says nothing. So the breakdown descends past any level that has only one phase, until
    it finds the first real split — that is the level where the work divides.

    Time the parent held but did not pass to a child becomes a trailing ``Other`` row, once
    it exceeds a tenth of a percent. The threshold lives here rather than in a renderer so
    the text report and the HTML one never disagree about whether that row exists.
    """
    prefix = _first_branching_prefix(tree)
    siblings = [p for p in tree if len(p) == len(prefix) + 1 and p[:-1] == prefix]
    parent = tree.get(prefix)
    unattributed = parent.self_ns if parent is not None else 0
    total = sum(tree[path].wall_ns for path in siblings) + unattributed
    if total <= 0:
        return []

    shares = [
        SiblingShare(name=path[-1], wall_ns=tree[path].wall_ns, total_ns=total)
        for path in sorted(siblings, key=lambda p: -tree[p].wall_ns)
    ]
    if unattributed > total * 0.001:
        shares.append(SiblingShare(name="Other", wall_ns=unattributed, total_ns=total))
    return shares


def wait_share(stats: PhaseStats) -> float:
    """``wait_ns`` as a percentage of ``wall_ns``, or 0.0 for a phase with no wall time.

    Pairs with wall time, never with self time: waiting inside a child still counts, so
    ``wait / self`` exceeds 100% for any phase that wraps a blocking call.
    """
    return 100.0 * stats.wait_ns / stats.wall_ns if stats.wall_ns else 0.0


def _share_rows(tree: PhaseTree, processes: int = 1) -> list[str]:
    """Format the pipeline breakdown derived by :func:`sibling_shares`.

    The wall column is marked when it sums several processes, so a figure larger than the
    run's own runtime reads as the aggregate it is rather than as a bug.
    """
    suffix = f" (Σ{processes} proc)" if processes > 1 else ""
    return [
        f"{share.name:<28}{share.percent:>7.1f}%{format_ns(share.wall_ns):>14}{suffix}"
        for share in sibling_shares(tree)
    ]


def _first_branching_prefix(tree: PhaseTree) -> PhasePath:
    """Descend while each level holds exactly one phase; return the first that branches."""
    prefix: PhasePath = ()
    while True:
        children = [p for p in tree if len(p) == len(prefix) + 1 and p[:-1] == prefix]
        if len(children) != 1:
            return prefix
        prefix = children[0]


def _dominant_rows(tree: PhaseTree, limit: int = 6) -> list[str]:
    """The phases holding the most *self* time — where the work actually happens.

    Ranked by self time rather than wall time, so a wrapper phase that merely contains its
    children never outranks the child doing the work.

    A row derived from a sampled phase is prefixed with ``~``: its numbers are scaled
    estimates, and every other number in this report is measured. A row whose phase declared
    ``async_work=True`` is prefixed with ``†``: its wall time is submission time, not the cost
    of the work it started. A phase that is both carries both marks.
    """
    ranked = sorted(tree.items(), key=lambda item: -item[1].self_ns)
    rows = [f"{'DOMINANT PHASES':<28}{'self':>12}{'wait':>8}{'p50':>10}{'p99':>10}"]
    for path, stats in ranked[:limit]:
        if not path or stats.self_ns <= 0:
            continue
        mark = _marks_of(stats)
        rows.append(
            f"{mark}{_label(path):<{28 - len(mark)}}{format_ns(stats.self_ns):>12}"
            f"{wait_share(stats):>7.0f}%"
            f"{format_ns(stats.hist.quantile(0.5)):>10}"
            f"{format_ns(stats.hist.quantile(0.99)):>10}",
        )
        rows.extend(_counter_rows(stats.counters, stats.wall_ns, stats))
    if len(rows) <= 1:
        return []
    rows.extend(_sampling_note(tree))
    rows.extend(_async_note(tree))
    return rows


def _marks_of(stats: PhaseStats) -> str:
    """The prefix characters qualifying a row's numbers, in a stable order.

    Both marks say the same kind of thing — this figure is not the plain measurement it looks
    like — so they compose rather than override. The label field is narrowed by the width of
    whatever this returns, which keeps the columns aligned however many marks apply.
    """
    marks = "~" if stats.sample_stride else ""
    if stats.async_entries:
        marks += "†"
    return marks


def _sampling_note(tree: PhaseTree) -> list[str]:
    """Say outright which phases were estimated, and at what rate.

    The ``~`` prefix alone is a symbol a reader has to guess at. Naming the rate is what makes
    the distinction actionable: a phase measured one entry in a hundred is a different kind of
    number from the rest of the report, not a slightly noisier one.
    """
    sampled = sorted(
        {"/".join(path): stats.sample_stride
         for path, stats in tree.items() if stats.sample_stride}.items(),
    )
    if not sampled:
        return []
    return [
        "",
        "  ~ = estimated from a sample, not measured. Totals are scaled by the rate:",
        *(f"      {format_label(name, 23):<24}1 entry in {stride:,}" for name, stride in sampled),
    ]


def _async_note(tree: PhaseTree) -> list[str]:
    """Say what a ``†`` row's wall time actually measured, and how to measure the other thing.

    The reading this prevents is the expensive one: a phase around an unsynchronised device
    submission holds most of a process's time at a plausible per-call latency, and every part
    of that is true except the implication that the device was busy for it. Naming the
    remedy in the note matters as much as the mark — ``sync=True`` is what turns the
    submission time into a device time, and it is not obvious from the symbol.
    """
    marked = sorted(
        {"/".join(path): (stats.async_entries, stats.calls)
         for path, stats in tree.items() if stats.async_entries}.items(),
    )
    if not marked:
        return []
    rows = [
        "",
        "  † = wall time excludes un-awaited device work (async_work=True). This is",
        "      submission time, not device compute. Re-run that phase with sync=True to",
        "      attribute the device time to it:",
    ]
    for name, (async_entries, calls) in marked:
        # A phase entered both ways is only partly submission time, and saying which part is
        # the difference between "this number is wrong" and "this number is a mixture".
        share = "" if async_entries >= calls else f" of {calls:,}"
        rows.append(f"      {format_label(name, 23):<24}{async_entries:,}{share} entries")
    return rows


def _iteration_rows(tree: PhaseTree) -> list[str]:
    """Quantiles for the repeating outer phase, whatever it is called here.

    The deepest-called top-level phase with the most entries is treated as the loop.
    """
    candidates = [
        (path, stats) for path, stats in tree.items() if len(path) == 1 and stats.calls > 2
    ]
    if not candidates:
        return []
    path, stats = max(candidates, key=lambda item: item[1].calls)
    mean = stats.wall_ns / stats.calls
    return [
        "",
        f"{path[0].upper()}S  ({stats.calls} entries)",
        f"  mean {format_ns(mean):>10}   p50 {format_ns(stats.hist.quantile(0.5)):>10}"
        f"   p95 {format_ns(stats.hist.quantile(0.95)):>10}"
        f"   p99 {format_ns(stats.hist.quantile(0.99)):>10}",
    ]


def _counter_rows(
    counters: dict[str, int],
    wall_ns: int,
    stats: PhaseStats | None = None,
) -> list[str]:
    """Work counters, their rate per second of the phase's wall time, and their spread.

    The ``io_*`` counters are skipped: they hold bytes, not work units, so a "per each"
    figure would be nonsense. They are rendered by :func:`_exact_io_block` instead.

    Every column is separated by a literal space rather than by trusting its width. A number
    is never truncated to fit — that would print a wrong one — so a field it overflows pushes
    the rest of the row right instead of running into it. A fast counter on a short phase did
    exactly that: 64 entries at 19,161,676.6/s rendered as ``6419,161,676.6/s``.
    """
    seconds = wall_ns / 1e9
    rows = []
    for name, total in sorted(counters.items()):
        if name.startswith("io_"):
            continue
        rate = total / seconds if seconds else 0.0
        per_unit = wall_ns / total if total else 0.0
        rows.append(
            f"    + {format_label(name, 21):<22}{total:>9,} "
            f"{rate:>11,.1f}/s {format_ns(per_unit):>8}/ea"
            f"{_counter_spread(name, stats)}",
        )
    return rows


def _counter_spread(name: str, stats: PhaseStats | None) -> str:
    """Render one counter's per-call range, or nothing when it never varied usefully.

    A single observation has no spread worth printing, and a counter always called with the
    same amount says so most compactly as ``always n`` — which is the whole finding when that
    amount is a configured cap. Anything else prints the range, because the distance between
    the ends is what separates "capped" from "bursty".
    """
    if stats is None:
        return ""
    low = stats.counter_min.get(name)
    high = stats.counter_max.get(name)
    if low is None or high is None:
        return ""
    if low == high:
        return f"  always {low:,}"
    return f"  {low:,}..{high:,}"


def _lifecycle_block(aligned: AlignedTrace | None) -> str:
    """Where a request's time actually went, between submission and response.

    The block a queue-driven pipeline is instrumented for. A single ``queue_wait`` total is
    correct and unactionable, because it fuses intervals whose remedies point in opposite
    directions — more batching, less batching, a cheaper model, fewer hops. Splitting it names
    which one you have.
    """
    if aligned is None:
        return ""
    channels = lifecycle_segments(aligned)
    if not channels:
        return ""

    lines = ["", "REQUEST LIFECYCLE", _RULE]
    for channel in sorted(channels):
        segments = channels[channel]
        total = sum(segment.total_ns for segment in segments)
        if total <= 0:
            continue
        requests = max(segment.count for segment in segments)
        lines.append(f"{format_label(channel, 27):<28}{format_ns(total):>12}  ({requests:,} req)")
        for index, segment in enumerate(segments):
            branch = "└─" if index == len(segments) - 1 else "├─"
            share = 100.0 * segment.total_ns / total
            lines.append(
                f"    {branch} {format_label(segment.name, 25):<26}"
                f"{format_ns(segment.total_ns):>10}{share:>6.0f}%"
                f"{format_ns(segment.mean_ns):>10}/ea",
            )
    if len(lines) <= 3:
        return ""
    return "\n".join(lines)


def _exact_io_block(tree: PhaseTree) -> str:
    """Phases opened with ``io=True``, whose bytes were measured at their own boundaries.

    Unlike the sampled block below, these numbers carry no attribution ambiguity: the
    counters were read on entry and exit of that exact phase. This is the block to read
    when asking which phase is I/O-bound.
    """
    rows = []
    for path, stats in sorted(tree.items(), key=lambda item: -_exact_io_total(item[1])):
        if _exact_io_total(stats) <= 0:
            continue
        rows.extend(_exact_io_rows(path, stats))
    if not rows:
        return ""
    return "\n".join(["", "I/O BY PHASE (measured exactly)", _RULE, *rows])


def _exact_io_rows(path: PhasePath, stats: PhaseStats) -> list[str]:
    """One row of disk traffic, plus a second naming what the page cache served.

    Disk bytes lead because they are what costs time on a saturated device. The cache line
    only appears when it changes the reading — without it a warm dataset looks like a phase
    that does no I/O at all, when in fact it is moving gigabytes out of RAM.
    """
    read = stats.counters.get("io_read_bytes", 0)
    write = stats.counters.get("io_write_bytes", 0)
    seconds = stats.wall_ns / 1e9
    rate = format_bytes((read + write) / seconds) + "/s" if seconds else "-"
    rows = [
        f"  {format_label('/'.join(path), 25):<26}"
        f"r {format_bytes(read):>10}   w {format_bytes(write):>10}{rate:>14}",
    ]
    cached = max(0, stats.counters.get("io_read_chars", 0) - read)
    if cached >= _CACHE_NOISE_FLOOR:
        rows.append(f"{'':<28}+ {format_bytes(cached)} read from page cache")
    return rows


def _exact_io_total(stats: PhaseStats) -> int:
    """Every byte the phase moved at either layer, so a warm read still ranks."""
    counters = stats.counters
    return max(
        counters.get("io_read_bytes", 0) + counters.get("io_write_bytes", 0),
        counters.get("io_read_chars", 0) + counters.get("io_write_chars", 0),
    )


def _io_block(analysis: SampleAnalysis) -> str:
    """Bytes moved, the phases that moved them, and a sparkline of when."""
    totals = analysis.totals
    moved = totals.read_bytes or totals.write_bytes or totals.read_chars or totals.write_chars
    if not moved:
        # Still render when every interval was discarded: "no I/O measured" and "I/O happened
        # but we could not read it" are opposite findings, and silence would show the first.
        return "\n".join(["", "I/O", _RULE, *_io_gap_note(analysis)]) if (
            analysis.io_gap_intervals
        ) else ""

    lines = ["", "I/O", _RULE]
    lines.append(f"{'Read (from disk)':<28}{format_bytes(totals.read_bytes):>14}"
                 f"{format_bytes(totals.read_rate) + '/s':>16}")
    if totals.cached_read_bytes:
        lines.append(f"{'Read (from page cache)':<28}{format_bytes(totals.cached_read_bytes):>14}")
    lines.append(f"{'Write':<28}{format_bytes(totals.write_bytes):>14}"
                 f"{format_bytes(totals.write_rate) + '/s':>16}")
    lines.extend(_io_phase_rows(analysis))
    lines.extend(_io_sparklines(analysis))
    lines.extend(_io_attribution_note(analysis))
    return "\n".join(lines)


def _io_phase_rows(analysis: SampleAnalysis, limit: int = 5) -> list[str]:
    """The phases during which most bytes moved."""
    ranked = sorted(
        analysis.io_by_phase.items(),
        key=lambda item: -(item[1].read_bytes + item[1].write_bytes + item[1].cached_read_bytes),
    )
    rows = [""] if ranked else []
    for phase, totals in ranked[:limit]:
        if not (totals.read_bytes or totals.write_bytes or totals.cached_read_bytes):
            continue
        cached = f"  cache {format_bytes(totals.cached_read_bytes)}" if (
            totals.cached_read_bytes
        ) else ""
        rows.append(
            f"  {format_label(phase, 25):<26}"
            f"r {format_bytes(totals.read_bytes):>10}   "
            f"w {format_bytes(totals.write_bytes):>10}{cached}",
        )
    return rows


def _io_attribution_note(analysis: SampleAnalysis) -> list[str]:
    """State the sampled block's resolution, and how much of it landed nowhere.

    The share is printed rather than implied: a run shorter than a few sample intervals can
    leave most of its bytes unattributed, and a reader who does not know that will read the
    per-phase rows above as a finding.
    """
    lines = [
        "",
        "  (bytes come from the OS process counters; attribution to a phase has a",
        "   resolution of one sample interval. Per-operation attribution needs eBPF.)",
    ]
    reads = analysis.unattributed_read_share
    writes = analysis.unattributed_write_share
    if max(reads, writes) > 0.05:
        lines.append(
            f"  {reads:.0%} of reads and {writes:.0%} of writes (syscall layer) moved while",
        )
        lines.append("   no phase was open — too coarse to attribute. Wrap those in io=True.")
    lines.extend(_io_gap_note(analysis))
    return lines


def _io_gap_note(analysis: SampleAnalysis) -> list[str]:
    """Declare intervals whose bytes were dropped because a counter read failed.

    Printed whenever it happens at all, with no threshold: the totals above become a floor
    rather than a measurement, and a reader has no other way to learn that.
    """
    gaps = analysis.io_gap_intervals
    if not gaps:
        return []
    share = gaps / analysis.io_intervals if analysis.io_intervals else 0.0
    return [
        "",
        f"  ⚠ {gaps} of {analysis.io_intervals} sample intervals ({share:.0%}) could not read",
        "   the process counters. Their bytes are excluded, so the totals above are a",
        "   lower bound.",
    ]


def _io_sparklines(analysis: SampleAnalysis) -> list[str]:
    """Show *when* the I/O happened; a total alone hides a stall at one moment."""
    if not analysis.read_series and not analysis.write_series:
        return []
    read_peak = format_bytes(max(analysis.read_series, default=0))
    write_peak = format_bytes(max(analysis.write_series, default=0))
    return [
        "",
        f"  read  {sparkline(analysis.read_series)}  peak {read_peak}/s",
        f"  write {sparkline(analysis.write_series)}  peak {write_peak}/s",
    ]


def _gpu_block(analysis: SampleAnalysis) -> str:
    """Sampled utilisation per device, this run's share of it, and peak allocator state."""
    if not analysis.has_gpu:
        return _gpu_absent_block(analysis)
    lines = ["", "GPU", _RULE]
    lines.extend(_gpu_utilisation_rows(analysis))
    if analysis.peak_cuda_reserved:
        lines.append(f"{'VRAM allocated (peak)':<28}{format_bytes(analysis.peak_cuda_alloc):>14}")
        lines.append(f"{'VRAM reserved (peak)':<28}{format_bytes(analysis.peak_cuda_reserved):>14}")
    lines.extend(_gpu_phase_rows(analysis))
    lines.append("")
    lines.extend(_gpu_footnote(analysis))
    return "\n".join(lines)


def _gpu_phase_rows(analysis: SampleAnalysis, limit: int = 6) -> list[str]:
    """Device utilisation while each phase was open, heaviest first.

    This is the table that makes an unsynchronised submission visible on the page instead of
    requiring a hand-rolled join: a phase named ``forward`` holding 97.9% of a server's time
    at 7% device utilisation is a contradiction anyone can see, and it was previously spread
    across two artifacts on two timebases.
    """
    rows = [
        usage for usage in analysis.gpu_by_phase.values()
        if usage.samples and usage.quantile(0.5) >= 0
    ]
    if not rows:
        return []
    rows.sort(key=lambda usage: -usage.quantile(0.5))
    lines = [
        "",
        f"{'GPU BY PHASE (sampled)':<28}{'p50':>7}{'p95':>7}{'VRAM':>12}{'samples':>10}",
    ]
    for usage in rows[:limit]:
        vram = format_bytes(usage.peak_cuda_reserved) if usage.peak_cuda_reserved else "n/a"
        lines.append(
            f"{format_label(usage.phase, 27):<28}"
            f"{usage.quantile(0.5):>6.0f}%{usage.quantile(0.95):>6.0f}%"
            f"{vram:>12}{usage.samples:>10,}",
        )
    return lines


def _gpu_absent_block(analysis: SampleAnalysis) -> str:
    """Say that no GPU data was collected, rather than saying nothing at all.

    Only when samples exist. An absent GPU section is indistinguishable from a run with no
    GPU in it, and a reader who cannot see the device sitting idle has no way to doubt a
    phase table that appears to show it saturated — which is exactly how an unsynchronised
    forward pass gets read as a GPU bottleneck. A run with no samples at all is a different
    situation and already says so elsewhere.
    """
    if not analysis.has_samples:
        return ""
    return "\n".join([
        "",
        "GPU",
        _RULE,
        "  No GPU data was collected. If this run used one, install nvidia-ml-py and",
        "  re-read the report without --no-samples; a phase around device work reports",
        "  submission time until something proves the device was busy.",
    ])


def _gpu_footnote(analysis: SampleAnalysis) -> list[str]:
    """Say what the numbers above are, and what they are not."""
    if analysis.gpu_devices:
        return [
            "  (busy is NVML's whole-device percentage — every process's kernels, not",
            "   just yours; 'this run' is the share NVML attributes to this run's own",
            "   pids. Neither is a compute-vs-wait split: for that, run with",
            "   backend='torch' and analyse the trace.)",
            *_gpu_phase_footnote(analysis),
        ]
    return [
        "  (utilisation is whole-device busy time from NVML, not a compute-vs-wait",
        "   split. For that, run with backend='torch' and analyse the trace.)",
        *_gpu_phase_footnote(analysis),
    ]


def _gpu_phase_footnote(analysis: SampleAnalysis) -> list[str]:
    """State the attribution rule for the per-phase rows, and its one sharp edge.

    The same caveat ``_exact_io_block`` carries for bytes, plus the async one: a phase that
    submits work without awaiting it will have its device time land under whichever *later*
    phase happens to synchronise. Left unsaid, the per-phase table would appear to refute the
    very mismeasurement it is there to expose.
    """
    if not analysis.gpu_by_phase:
        return []
    return [
        "",
        "  (per-phase rows attribute each 1 Hz sample to the phase open when it was",
        "   taken. A phase that submits device work without awaiting it may have that",
        "   work land under a later phase, so low utilisation on an async phase is",
        "   expected — and is the reason its wall time is submission time.)",
    ]


def _gpu_utilisation_rows(analysis: SampleAnalysis) -> list[str]:
    """One row per device, falling back to a single figure for pre-per-device sample files."""
    if not analysis.gpu_devices:
        if analysis.gpu_util_mean < 0:
            return []
        return [
            f"{'Utilisation (sampled)':<28}{analysis.gpu_util_mean:>7.1f}%",
            f"{'Idle (sampled)':<28}{100.0 - analysis.gpu_util_mean:>7.1f}%",
        ]

    rows = [f"{'':<28}{'busy':>7}  {'this run':>9}  {'idle':>7}"]
    for device in analysis.gpu_devices:
        ours = f"{device.ours_mean:>8.1f}%" if device.ours_mean >= 0 else f"{'n/a':>9}"
        idle = f"{100.0 - device.busy_mean:>6.1f}%" if device.busy_mean >= 0 else f"{'n/a':>7}"
        busy = f"{device.busy_mean:>6.1f}%" if device.busy_mean >= 0 else f"{'n/a':>7}"
        rows.append(f"{f'GPU {device.index}':<28}{busy}  {ours}  {idle}")
    return rows


def _memory_block(analysis: SampleAnalysis) -> str:
    """Resident memory, its trend, and which phase it grows under."""
    if not analysis.memory.last_rss:
        return ""
    memory = analysis.memory
    lines = ["", "MEMORY (RSS)", _RULE]
    lines.append(f"{'Current':<28}{format_bytes(memory.last_rss):>14}")
    lines.append(f"{'Peak':<28}{format_bytes(memory.peak_rss):>14}")
    lines.append(f"{'Growth over run':<28}{format_bytes(memory.growth_bytes):>14}")
    lines.append(f"{'Trend':<28}{format_bytes(memory.slope_bytes_per_s) + '/s':>14}")
    lines.extend(_memory_phase_rows(analysis))
    return "\n".join(lines)


def _memory_phase_rows(analysis: SampleAnalysis, limit: int = 3) -> list[str]:
    """The phases with the steepest upward RSS slope — the leak candidates."""
    ranked = sorted(
        analysis.memory_by_phase.items(), key=lambda item: -item[1].slope_bytes_per_s,
    )
    rows = []
    for phase, trend in ranked[:limit]:
        if trend.slope_bytes_per_s <= 0:
            continue
        rate = format_bytes(trend.slope_bytes_per_s) + "/s"
        rows.append(f"  growing under {format_label(phase, 23):<24}{rate:>14}")
    return [""] + rows if rows else []


def _backend_block(run: MergedRun) -> str:
    """Point at the heavy-profiler artifacts, which answer the questions this layer cannot."""
    artifacts = run.backend_artifacts()
    if not artifacts:
        return ""
    lines = ["", "BACKEND ARTIFACTS", _RULE]
    for entry in artifacts:
        lines.append(f"  {entry['backend']:<12}{entry['artifact']}")
    return "\n".join(lines)


def _losses_block(run: MergedRun) -> str:
    """Everything this report is *not* telling you: lost files, stale workers, other attempts.

    Kept as one block at the end, printed only when there is something to say, because a
    report that under-reports silently is worse than one that reports nothing.
    """
    sections = [
        _unreadable_rows(run),
        _superseded_rows(run),
        _degraded_rows(run),
    ]
    body = [line for section in sections for line in section]
    if not body:
        return ""
    return "\n".join(["", "CAVEATS", _RULE, *body])


def _unreadable_rows(run: MergedRun) -> list[str]:
    if not run.unreadable:
        return []
    rows = [f"{len(run.unreadable)} worker file(s) unreadable — their work is missing:"]
    rows.extend(f"  {path.name}" for path in run.unreadable)
    return rows


def _superseded_rows(run: MergedRun) -> list[str]:
    """Declare workers from an earlier attempt in the same directory, and exclude them."""
    if not run.superseded:
        return []
    attempts = sorted({worker.run_id for worker in run.superseded})
    return [
        f"{len(run.superseded)} worker file(s) belong to {len(attempts)} earlier attempt(s)",
        "  in this directory and are excluded from every total above:",
        *(f"  {attempt}" for attempt in attempts),
    ]


def _stale_workers(run: MergedRun) -> list[WorkerSnapshot]:
    """Workers whose last snapshot trails the run's by more than the staleness window.

    One definition, shared by the text report and ``report_as_dict``: a worker that stopped
    writing hours ago leaves a file that parses perfectly, so a caller reading the JSON needs
    to be told exactly what a reader of the text is told.
    """
    latest = max((w.written_at for w in run.workers), default=0.0)
    return [w for w in run.workers if latest - w.written_at > _STALE_AFTER_S]


def _degraded_rows(run: MergedRun) -> list[str]:
    """Name workers whose own snapshots were failing, and workers that stopped early.

    A worker whose flushes fail keeps a file that parses perfectly and is simply out of date,
    so staleness has to be derived here rather than trusted from the file itself.
    """
    failing = [w for w in run.workers if w.write_failures]
    stale = _stale_workers(run)
    latest = max((w.written_at for w in run.workers), default=0.0)
    rows: list[str] = []
    if failing:
        rows.append(f"{len(failing)} worker(s) reported failed snapshot writes:")
        rows.extend(f"  {w.label}  {w.write_failures} failure(s)" for w in failing[:8])
    if stale:
        rows.append(f"{len(stale)} worker(s) stopped writing well before the run ended:")
        rows.extend(
            f"  {w.label}  last wrote {format_ns((latest - w.written_at) * 1e9)} early"
            for w in stale[:8]
        )
    return rows


def _label(path: PhasePath) -> str:
    """Render a phase path compactly: the leaf, with its parent when that disambiguates."""
    if len(path) == 1:
        return path[0]
    return format_label(f"{path[-2]}/{path[-1]}", 27)
