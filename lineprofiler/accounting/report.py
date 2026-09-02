"""Plain-text rendering of a merged run.

The report is organised by *role* — learner, actor, inference server, or whatever this
architecture calls its processes — because a single global percentage is misleading the
moment several workers run concurrently. Sixteen actors doing self-play will always
dominate a global pie chart, whether or not self-play is the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from lineprofiler.accounting.analysis import (
    CpuUsage,
    SampleAnalysis,
    analyse_processes,
    format_bytes,
    sparkline,
)
from lineprofiler.accounting.findings import Finding, rank_findings
from lineprofiler.accounting.hardware import (
    format_capacity,
)
from lineprofiler.accounting.hardware import (
    total_vram as hardware_total_vram,
)
from lineprofiler.accounting.phasetree import PhasePath, PhaseStats, PhaseTree
from lineprofiler.accounting.provenance import source_of
from lineprofiler.accounting.snapshot import MergedRun, WorkerSnapshot, imbalance_of
from lineprofiler.accounting.trace import FLAG_DEVICE_SYNC
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
    blocks = [
        _header(run),
        _findings_block(rank_findings(aligned) if aligned is not None else []),
        _resources_block(analysis, run),
    ]
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
    # Findings are part of the document rather than a rendering of it: a CI gate asking
    # "did this run regress into a queue" needs the verdict, not the prose that explains it.
    aligned = _aligned_or_none(run)
    if aligned is not None:
        findings = rank_findings(aligned)
        if findings:
            document["findings"] = [_finding_as_dict(finding) for finding in findings]

    samples = run.samples_by_process()
    analysis = analyse_processes(samples) if samples else None
    if analysis is not None:
        document["resources"] = _resources_as_dict(analysis)
    # A separate key from ``resources``, which is already published and holds the io/memory/gpu
    # breakdown. This one answers "what did it run on and how does that scale", and appears
    # only when something is actually known.
    machine = _machine_as_dict(analysis, run)
    if machine:
        document["machine"] = machine
    return document


def _finding_as_dict(finding: Finding) -> dict[str, Any]:
    """One ranked finding as data, carrying the figure a gate would threshold on.

    ``cost_pct`` is share of the traced span and is the same denominator for every kind, which
    is what lets a gate compare an idle lane against a blocked phase at all.
    """
    return {
        "kind": finding.kind,
        "headline": finding.headline,
        "detail": finding.detail,
        "cost_pct": round(finding.cost_pct, 2),
        "anchor": finding.anchor,
        "lanes": list(finding.lanes),
    }


def _machine_as_dict(analysis: SampleAnalysis | None, run: MergedRun) -> dict[str, Any]:
    """Consumption and capacity as data, with the per-process denominator stated alongside.

    ``capacity_by_host`` stays keyed rather than flattened so a script can see a heterogeneous
    run for what it is, the same reason the text report prints one footnote line per host.
    """
    machine: dict[str, Any] = {}
    processes = max(len(run.workers), 1)
    if analysis is not None and (analysis.cpu.measured or analysis.memory.peak_rss):
        machine["used"] = {
            "processes": processes,
            "cpu_cores_peak": analysis.cpu.peak if analysis.cpu.measured else None,
            "cpu_cores_mean": analysis.cpu.mean if analysis.cpu.measured else None,
            "cpu_cores_max_process": analysis.cpu.max_process if analysis.cpu.measured else None,
            "rss_peak": analysis.memory.peak_rss,
            "rss_max_process": analysis.peak_rss_max_process,
            "vram_peak": analysis.peak_cuda_alloc,
            # The allocator's peak and the device's, under names that say which is which. A
            # capacity question ("does another worker fit?") is only answerable from the
            # second, because only it counts the per-process CUDA context.
            "vram_held_peak": (
                analysis.peak_cuda_process if analysis.cuda_process_measured else None
            ),
            "per_process": {
                "cpu_cores": analysis.cpu.peak / processes if analysis.cpu.measured else None,
                "rss": analysis.memory.peak_rss / processes,
                "vram": analysis.peak_cuda_alloc / processes,
                "vram_held": (
                    analysis.peak_cuda_process / processes
                    if analysis.cuda_process_measured else None
                ),
            },
        }
    by_host = run.hardware_by_host
    if by_host:
        machine["capacity_by_host"] = by_host
    return machine


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
            # None rather than 0 when unread: a consumer must be able to tell "this run held
            # no device memory" from "this driver would not say".
            "peak_cuda_process": (
                analysis.peak_cuda_process if analysis.cuda_process_measured else None
            ),
            "devices": [
                {"index": d.index, "busy_mean": d.busy_mean, "ours_mean": d.ours_mean}
                for d in analysis.gpu_devices
            ],
        },
    }


UNUSABLE = "n/a"
"""What a renderer prints where a duration could not be a duration.

Deliberately not ``0ns``, which is a real reading meaning "too fast to measure" — the same
distinction ``UNMEASURED = -1`` makes in the trace buffer and ``IoSnapshot.available`` makes
for byte counters. A reader who sees this knows to distrust that one figure; a reader who
sees a zero has been told something false.

Deliberately not ``"?"`` either, which this report already uses for an unknown *host* and
``compare`` uses for a thin sample. ``Runtime ?`` printed directly above ``Host ?`` reads as
one kind of gap in two places, and the two mean different things. ``n/a`` is also what the
wait column already prints for an unmeasured share, so the page keeps one vocabulary.
"""


def format_ns(value: float) -> str:
    """Render a nanosecond duration with a unit that keeps three significant digits.

    A non-finite or negative input returns :data:`UNUSABLE` rather than raising or printing a
    duration that cannot exist. Both are reachable from a worker file that satisfies every
    guard in ``_read_worker``: ``float("inf")`` and ``float("nan")`` *are* floats, and
    ``json.dumps`` writes ``Infinity``/``NaN`` by default while ``json.loads`` reads them
    back, so the writer round-trips a broken clock reading silently. ``int()`` then raised
    from the first line of the header, which cost the whole report — every other worker
    included — for one bad file. Negative durations arrive the same way, from a wall clock
    stepped backwards mid-run by an NTP correction.
    """
    if not isfinite(value) or value < 0:
        return UNUSABLE
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


def format_rate(value: float) -> str:
    """Render a per-second rate, dropping to exponent form where the digits stop informing.

    The grouped form is what a reader expects and it is kept for every ordinary magnitude.
    Past a quadrillion the digits are no longer telling anyone anything — a thirty-digit
    figure quoted to the tenth is harder to read than the exponent and implies a precision the
    measurement does not have. The threshold is well above any real work rate, so this is the
    overflow path, not the common one.
    """
    if value >= 1e15:
        return f"{value:.2e}"
    return f"{value:,.1f}"


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

    Control characters are replaced before any of that. A phase name is user data, and a
    newline in one broke a single row into four — three of them carrying no numbers, which
    reads as three phases that do not exist. Escaping is no help here: these characters are
    legal in the output and destroy the layout anyway. Replacing rather than dropping keeps
    the name's length honest, so two names differing only by a control character stay
    distinguishable on the page.
    """
    text = _printable(text)
    if len(text) <= width:
        return text
    return "…" + text[-(width - 1):]


def _printable(text: str) -> str:
    """Replace characters that would break a row into lines or corrupt a cell.

    Covers the C0 and C1 ranges plus the line and paragraph separators, which a terminal and
    a browser disagree about but neither renders as a character in a table cell.
    """
    return "".join(
        "\ufffd" if (ch < " " or "\x7f" <= ch <= "\x9f" or ch in "\u2028\u2029") else ch
        for ch in text
    )


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
    lines.extend(_unusable_runtime_warning(runtime))
    lines.extend(_self_nesting_warning(run.tree))
    lines.extend(_excluded_workers_warning(run))
    return "\n".join(lines)


def _self_nesting_warning(tree: PhaseTree) -> list[str]:
    """Call out a phase recorded as nested inside itself, which usually is not nesting.

    Phase stacks are per thread. Asyncio tasks share a thread, so two tasks inside one phase
    put it on the stack twice and it is stored as its own child — every level claiming the
    full duration, and the outermost reporting a single entry for however many requests were
    served. The profiler warns the author at the point it happens, but a report is read by
    whoever opens the file, usually elsewhere and later, with no terminal in sight. This is
    the one tree shape that must not be read at face value, so it is named here.

    True recursion produces the same shape and is measured correctly, which is why this is
    worded as the likelier cause rather than as a verdict.
    """
    repeated = sorted({
        path[-1] for path in tree if len(path) > 1 and path[-1] in path[:-1]
    })
    if not repeated:
        return []
    names = ", ".join(repr(name) for name in repeated[:3])
    more = f" (and {len(repeated) - 3} more)" if len(repeated) > 3 else ""
    return [
        f"WARNING  {names}{more} appears nested inside itself. If these are asyncio tasks",
        "         sharing a thread, that is concurrency recorded as nesting: each level"
        " claims",
        "         the whole duration and the outermost counts one entry per batch of"
        " requests.",
        "         Genuine recursion produces the same shape and is measured correctly.",
    ]


def _unusable_runtime_warning(runtime: float) -> list[str]:
    """Say why the runtime is missing, and what on the page is unaffected by it.

    ``n/a`` states that a figure is unavailable without saying which of two very different
    things happened, and the reader needs that to know how much of the rest to trust. A
    worker's timestamps come from the wall clock, which an NTP correction can step backwards
    mid-run, and a non-finite one round-trips through JSON silently; neither touches the phase
    tree, whose durations are ``perf_counter`` deltas. So the numbers below the header are
    still measurements, and saying so is the point of the line — without it a reader who
    distrusts the runtime has no reason to trust anything under it either.
    """
    if isfinite(runtime) and runtime >= 0:
        return []
    return [
        "WARNING  the run's wall clock is unusable (a timestamp is negative or non-finite),",
        "         so the runtime above could not be computed. Phase totals are unaffected:",
        "         they are perf_counter deltas and do not come from the wall clock.",
    ]


def _excluded_workers_warning(run: MergedRun) -> list[str]:
    """Warn beside the process count when most of the workers present were excluded.

    CAVEATS already lists superseded attempts, but it prints at the foot of the report —
    below the findings, the role blocks and every total, all of which were computed without
    those workers. When the excluded files outnumber the kept ones, the reader is looking at a
    minority of their run under a header that says "Processes 1", and the conclusions above
    the caveat are conclusions about one worker presented as conclusions about the job.

    The common cause is not a rerun at all: workers that each construct their own ``Profiler``
    without inheriting ``LINEPROFILER_RUN_ID`` get one attempt id each, so a healthy four-way
    job reads as four competing attempts. Naming that here is what turns a silently narrowed
    report into a one-line fix.
    """
    if not run.superseded or len(run.superseded) <= len(run.workers):
        return []
    attempts = len({worker.run_id for worker in run.superseded})
    return [
        f"WARNING  {len(run.superseded)} of {len(run.superseded) + len(run.workers)} worker "
        f"file(s) are excluded as {attempts} earlier attempt(s) — see CAVEATS.",
        "         If these ran together, they each generated their own run id: pass the same "
        "run_id=",
        "         to every worker, or let them inherit LINEPROFILER_RUN_ID from the parent.",
    ]


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


def _resources_block(analysis: SampleAnalysis, run: MergedRun) -> str:
    """What the run consumed, what the machines had, and how both scale per worker.

    Placed first because it is the frame every later number is read against. A phase costing
    "675 ms over 2 processes" means something different on a 128-core node than on a laptop,
    and two reports from two servers cannot be compared at all without it.

    Test specifically:
        - a run with no samples and no hardware renders nothing rather than a header of zeros
        - a run whose samples carry no CPU reading omits the CPU rows and says why
        - worker files written before hardware was recorded omit the available column
        - two hosts with different capacity produce one footnote line each
    """
    capacity = pooled_capacity(run)
    rows = _resource_rows(analysis, run, capacity)
    # Capacity alone still earns the section: a run recorded with sampling off knows nothing
    # about its consumption and everything about the machine, and naming that machine is what
    # makes its timings comparable against another server's.
    if not rows and not capacity:
        return ""
    lines = ["", "RESOURCES", _RULE]
    if rows:
        lines.append(f"{'':<20}{'used':>12}{'available':>14}{'per proc':>14}")
        lines.extend(rows)
    return "\n".join([*lines, *_resource_notes(analysis, run, measured=bool(rows))])


def _resource_rows(
    analysis: SampleAnalysis,
    run: MergedRun,
    capacity: dict[str, Any],
) -> list[str]:
    """One row per resource that has something to report, in CPU / RAM / VRAM order."""
    processes = max(len(run.workers), 1)
    rows: list[str] = []
    if analysis.cpu.measured:
        rows.extend(_cpu_rows(analysis.cpu, capacity, processes))
    if analysis.memory.peak_rss:
        rows.append(_resource_row(
            "RAM  peak RSS",
            format_bytes(analysis.memory.peak_rss),
            format_bytes(capacity["ram_total"]) if capacity.get("ram_total") else "",
            format_bytes(analysis.memory.peak_rss / processes),
        ))
    vram = hardware_total_vram(capacity.get("gpus", []))
    if analysis.peak_cuda_alloc:
        rows.append(_resource_row(
            "VRAM peak alloc",
            format_bytes(analysis.peak_cuda_alloc),
            format_bytes(vram) if vram else "",
            format_bytes(analysis.peak_cuda_alloc / processes),
        ))
    # Beside it, never instead of it. The allocator figure is the right answer to "how big are
    # my tensors"; this one is the right answer to "will another worker fit on the card", and
    # only this one includes the per-process CUDA context. Replacing the older row would also
    # silently change what every archived report is being compared against.
    if analysis.cuda_process_measured:
        rows.append(_resource_row(
            "VRAM peak held",
            format_bytes(analysis.peak_cuda_process),
            format_bytes(vram) if vram else "",
            format_bytes(analysis.peak_cuda_process / processes),
        ))
    gpus = capacity.get("gpus") or []
    if gpus:
        rows.append(f"{'GPU  devices':<20}{'':>12}{len(gpus):>14}")
    return rows


def _cpu_rows(cpu: CpuUsage, capacity: dict[str, Any], processes: int) -> list[str]:
    """Peak and mean core-equivalents. Peak carries the capacity, since that is the ceiling."""
    # The affinity figure when the job was constrained, the box's cores otherwise: on a shared
    # node the machine's total overstates the headroom this run actually had.
    available = capacity.get("cpu_affinity") or capacity.get("cpu_cores")
    return [
        _resource_row(
            "CPU  peak",
            f"{cpu.peak:.1f} cores",
            f"{available} cores" if available else "",
            f"{cpu.peak / processes:.2f}",
        ),
        _resource_row(
            "CPU  mean",
            f"{cpu.mean:.1f} cores",
            _percent_of(cpu.mean, available),
            f"{cpu.mean / processes:.2f}",
        ),
    ]


def _resource_row(label: str, used: str, available: str, per_process: str) -> str:
    """One aligned row. An empty ``available`` leaves the column blank rather than zeroed."""
    return f"{label:<20}{used:>12}{available:>14}{per_process:>14}"


def _percent_of(used: float, available: float | None) -> str:
    """``used`` as a share of ``available``, or ``""`` when there is nothing to divide by.

    Absent capacity must suppress the figure entirely. Dividing by a missing field, or
    substituting a default for it, invents a utilisation the run never demonstrated.
    """
    if not available:
        return ""
    return f"({used / available * 100:.0f}% of box)"


def pooled_capacity(run: MergedRun) -> dict[str, Any]:
    """Sum every participating host's capacity into one machine-shaped dict.

    Summed across the hosts that actually ran workers, never one node's figures multiplied by
    the node count: a run spanning a fat node and a thin one has neither node's capacity, and
    guessing which to scale would misstate the headroom in whichever direction it guessed.
    """
    pooled: dict[str, Any] = {}
    gpus: list[dict[str, Any]] = []
    for hardware in run.hardware_by_host.values():
        for key in ("cpu_cores", "cpu_threads", "cpu_affinity", "ram_total"):
            value = hardware.get(key)
            if value:
                pooled[key] = pooled.get(key, 0) + int(value)
        gpus.extend(hardware.get("gpus") or [])
    if gpus:
        pooled["gpus"] = gpus
    return pooled


def _resource_notes(analysis: SampleAnalysis, run: MergedRun, measured: bool = True) -> list[str]:
    """The denominators, the machines, and any resource that went unmeasured.

    ``measured`` is False when the section has no consumption rows at all, which suppresses
    the per-process denominator: there is nothing above it for that denominator to divide.
    """
    notes: list[str] = []
    by_host = run.hardware_by_host
    for host, hardware in list(by_host.items())[:4]:
        summary = format_capacity(hardware)
        if summary:
            notes.append(f"  {host}: {summary}")
    if len(by_host) > 4:
        notes.append(f"  +{len(by_host) - 4} further hosts")
    if not by_host:
        notes.append("  (machine capacity was not recorded for this run)")

    processes = max(len(run.workers), 1)
    if measured:
        roles = ", ".join(f"{role} x{len(run.workers_of(role))}" for role in run.roles)
        suffix = f" ({roles})" if roles else ""
        notes.append(f"  per-proc figures are over {processes} process(es){suffix}")
    if analysis.memory.peak_rss and analysis.peak_rss_max_process:
        notes.append(
            f"  heaviest process held {format_bytes(analysis.peak_rss_max_process)} RSS"
            f" against a {format_bytes(analysis.memory.peak_rss / processes)} mean",
        )
    # The CPU counterpart of the RSS skew line. `peak` sums every process at its own peak, which
    # is a worst case that no instant need have contained; the skew is what says whether one
    # process drove it or the load was spread. It was computed and carried into report_as_dict
    # but never rendered, so the block printed the alarming figure without the one beside it that
    # tells a reader how to size a job.
    if analysis.cpu.measured and analysis.cpu.max_process:
        notes.append(
            f"  heaviest process peaked at {analysis.cpu.max_process:.2f} cores"
            f" against a {analysis.cpu.peak / processes:.2f} mean",
        )
    notes.extend(_vram_notes(analysis, run))
    # Only worth saying when the run sampled at all: a run with sampling switched off is not
    # missing a CPU capability, it declined to measure anything.
    if measured and not analysis.cpu.measured:
        notes.append("  (no CPU readings in this run's samples; install psutil to record them)")
    return [""] + notes if notes else []


def _vram_notes(analysis: SampleAnalysis, run: MergedRun) -> list[str]:
    """What the two VRAM rows measure, and any role holding a context it never uses."""
    if not analysis.cuda_process_measured:
        return []
    notes = [
        "  VRAM peak alloc is the torch allocator (tensors only); peak held is what the",
        "  device reports for this run's pids, which includes each process's CUDA context",
    ]
    return notes + _idle_context_notes(run)


def idle_context_roles(run: MergedRun) -> list[tuple[str, int, float]]:
    """Roles holding VRAM with no allocator activity, as ``(role, processes, bytes each)``.

    The signature of profiling changing what it measures. ``phase(sync=True)`` used to call
    ``torch.cuda.synchronize()`` in every process where a device was visible, and that call is
    what creates a ~414 MiB primary context — so a CPU-only actor paid for one, invisibly,
    because the allocator figure beside it never sees a context. Four actors cost 1.7 GB of a
    40 GB card; the runbook's own 32-actor profile would cost ~13 GB.

    Derived here rather than in either renderer so the text report and the HTML page cannot
    disagree about which roles are affected.
    """
    rows: list[tuple[str, int, float]] = []
    for role in run.roles:
        idle = [worker for worker in run.workers_of(role) if _holds_an_idle_context(worker)]
        if not idle:
            continue
        held = sum(_peak_device_vram(worker) for worker in idle)
        rows.append((role, len(idle), held / len(idle)))
    return rows


def _idle_context_notes(run: MergedRun) -> list[str]:
    """:func:`idle_context_roles`, wrapped to the report's column."""
    notes: list[str] = []
    for role, processes, each in idle_context_roles(run):
        notes.extend([
            f"  {processes} {role} process(es) hold {format_bytes(each)} of VRAM each with "
            f"no allocator activity:",
            "  a CUDA context in a process doing no GPU work. Pass cuda_sync=False to",
            "  that role's Profiler if it holds no model.",
        ])
    return notes


def _holds_an_idle_context(worker: WorkerSnapshot) -> bool:
    """Whether this worker holds device memory it never allocated a tensor in."""
    return _peak_device_vram(worker) > 0 and not any(
        sample.cuda_reserved for sample in worker.samples
    )


def _peak_device_vram(worker: WorkerSnapshot) -> int:
    """The most VRAM the device reported this worker holding, or ``0`` when never measured."""
    return max((s.cuda_proc_used for s in worker.samples if s.cuda_proc_used >= 0), default=0)


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
        return _device_wait_rows(role, aligned)
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
    """The role's spans that spent most of their time off-CPU waiting on something else.

    Overlapping and nested spans are harmless here: :func:`concurrent_activity` merges its
    windows before measuring, so the same blocked microsecond cannot be counted twice however
    deeply the phases nest.

    A phase that drained a CUDA queue is excluded, and so is any phase that encloses one: the
    first is off-CPU because the device is running *its own* work, and the second inherits that
    wait from its child. Asking who else was busy during either answers a question nobody asked
    and prints the answer as a verdict — "this is a stall, not a queue", about a backward pass.
    Nothing is lost by dropping the enclosing phase: a parent that also waits on something real
    does so inside a child that is still in this list, carrying the same wait at the depth that
    can be attributed.
    """
    enclosing = _phases_around_a_device_sync(aligned, role)
    return [
        span for span in aligned.spans
        if span.role == role and span.cpu_measured and span.duration_ns > 0
        and span.wait_ns > span.duration_ns * 0.5
        and not span.flags & FLAG_DEVICE_SYNC
        and span.path not in enclosing
    ]


def _phases_around_a_device_sync(aligned: AlignedTrace, role: str) -> set[PhasePath]:
    """Paths of this role's phases that contain a ``sync=True`` phase.

    Containment by path prefix, which is exact for a named phase: its path *is* its call stack.
    """
    enclosing: set[PhasePath] = set()
    for span in aligned.spans:
        if span.role != role or not span.flags & FLAG_DEVICE_SYNC:
            continue
        enclosing.update(span.path[:depth] for depth in range(1, len(span.path)))
    return enclosing


def _device_wait_rows(role: str, aligned: AlignedTrace) -> list[str]:
    """Say that a role's off-CPU time was the device, when that is all it was.

    Reached when nothing is left after the exclusion above. Saying nothing would be the safer
    silence, but a role with no concurrency line beside a `wait 100%` column reads as a gap in
    the report rather than as an answer.
    """
    device = [
        span for span in aligned.spans
        if span.role == role and span.flags & FLAG_DEVICE_SYNC and span.wait_ns > 0
    ]
    if not device:
        return []
    return [
        "",
        f"  {role}'s off-CPU time is device work on sync=True phases, not a wait on a peer.",
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

    The label goes through :func:`format_label` like every other in this report. A bare
    ``{name:<28}`` pads a short name but does not bound a long one, so a phase named from data
    — a path, a URL, a serialised config — printed a row as wide as the name and pushed its own
    percentage and wall time off to a column nobody scans. Truncation here is what keeps the
    share column a column.
    """
    suffix = f" (Σ{processes} proc)" if processes > 1 else ""
    return [
        f"{format_label(share.name, 27):<28}{share.percent:>7.1f}%"
        f"{format_ns(share.wall_ns):>14}{suffix}"
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

    ``entries`` is how many times the phase was entered — ``PhaseStats.calls``, which every
    phase increments on the way out whether or not anyone asked for it. Printing it makes a
    ``count()`` whose amount is always 1 redundant, which was the common shape of one: before
    this column a counter was the only way to get the number onto the page at all. What a
    counter still earns is an amount that *varies*, which nothing here can infer.
    """
    ranked = sorted(tree.items(), key=lambda item: -item[1].self_ns)
    rows = [
        f"{'DOMINANT PHASES':<25}{'entries':>7}{'self':>12}"
        f"{'wait':>8}{'p50':>10}{'p99':>10}",
    ]
    for path, stats in ranked[:limit]:
        if not path or stats.self_ns <= 0:
            continue
        mark = _marks_of(stats)
        rows.append(
            f"{mark}{_label(path, 25 - len(mark)):<{25 - len(mark)}}{stats.calls:>7,}"
            f"{format_ns(stats.self_ns):>12}"
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
    # The next question after "this is submission time" is "why is submission slow", and the
    # measurement that answers it is one the profiler cannot run for you — but it can name it,
    # and it costs one line beside the counter rows the reader is already looking at.
    rows.extend([
        "      If a phase's cost does not scale with its batch counter, it is launch-bound:",
        "      the time is per-kernel launch overhead, not per-element work. Compare its wall",
        "      time at batch 1 against a large batch; flat means fewer, bigger launches (a",
        "      captured CUDA graph) is the fix, not a faster kernel.",
    ])
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

    The *count* is therefore printed in full however large it is. The *rate* is not the same
    kind of number: it is derived, and past a point its digits stop being information. A
    counter of 2**70 on a microsecond phase rendered a thirty-digit per-second figure claimed
    to the tenth. :func:`format_rate` falls back to three significant digits there, which is
    the convention :func:`format_ns` already sets for every duration in this report.
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
            f"{format_rate(rate):>11}/s {format_ns(per_unit):>8}/ea"
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


def _findings_block(findings: list[Finding]) -> str:
    """What is wrong with this run, ranked, before any of the numbers behind it.

    The same derivation the HTML timeline uses, so the terminal and the page can never
    disagree about what the bottleneck was — the reason this lives in ``findings.py`` rather
    than in either renderer.

    Silent on a run with no trace: findings come from spans, and a phase tree alone cannot say
    who was waiting for whom. That is a real limit rather than an omission, and inventing a
    weaker finding from totals would put a sentence at the top of the report that the rest of
    it could not support.
    """
    if not findings:
        return ""
    lines = ["", "FINDINGS", _RULE]
    for index, finding in enumerate(findings, start=1):
        lines.append(f"{index}. {finding.headline}")
        # Wrapped rather than truncated: the detail carries the queue-versus-stall verdict,
        # which is the half a reader cannot reconstruct from the headline.
        lines.extend(f"   {line}" for line in _wrapped(finding.detail, _WIDTH - 3))
    return "\n".join(lines)


def _wrapped(text: str, width: int) -> list[str]:
    """Break ``text`` on spaces to fit ``width``, so the block keeps the report's column."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


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


def _label(path: PhasePath, field: int = 25) -> str:
    """Render a phase path compactly: the leaf, with its parent when that disambiguates.

    ``field`` is the column width the caller will pad into, and the label is truncated one
    column narrower so the padding always leaves a gap before the next heading. The caller
    passes a field already reduced by the width of any ``~``/``†`` marks it prefixes: those
    marks eat into the same columns, and truncating to a fixed width regardless of them
    pushed a two-mark row one column wider than every other.

    At the default the limit is 24, which is what an ordinary ``parent/leaf`` pair costs:
    ``iteration/backpropagate``, ``select/score_children`` and ``learner/optimizer_step`` all
    fit. No width fits every name — ``trainer/gradient_accumulation`` is 29 — but this is where
    the returns stop: a truncated label is the one column a reader cannot reconstruct from the
    others, and the names that still exceed it are long enough that their leaf is the only part
    worth printing anyway.
    """
    text = path[0] if len(path) == 1 else f"{path[-2]}/{path[-1]}"
    return format_label(text, field - 1)
