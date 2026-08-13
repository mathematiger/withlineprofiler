"""Plain-text rendering of a merged run.

The report is organised by *role* — learner, actor, inference server, or whatever this
architecture calls its processes — because a single global percentage is misleading the
moment several workers run concurrently. Sixteen actors doing self-play will always
dominate a global pie chart, whether or not self-play is the bottleneck.
"""

from __future__ import annotations

from lineprofiler.accounting.analysis import (
    SampleAnalysis,
    analyse_processes,
    format_bytes,
    sparkline,
)
from lineprofiler.accounting.phase import PhasePath, PhaseStats, PhaseTree
from lineprofiler.accounting.snapshot import MergedRun, imbalance_of

_WIDTH = 62
_RULE = "─" * _WIDTH

_CACHE_NOISE_FLOOR = 64 * 1024
"""Below one readahead window, cached reads are interpreter noise rather than your data."""


def render(run: MergedRun) -> str:
    """Return the full text report for a merged run.

    Test specifically:
        - golden-file comparison against a fixed synthetic run
        - a run with no phases and no samples renders without raising
        - unreadable workers are named in the output rather than silently dropped
        - a run with two roles renders a separate block for each
    """
    analysis = analyse_processes(run.samples_by_process())
    blocks = [_header(run)]
    for role in run.roles:
        blocks.append(_role_block(run, role))
    blocks.append(_exact_io_block(run.tree))
    blocks.append(_io_block(analysis))
    blocks.append(_gpu_block(analysis))
    blocks.append(_memory_block(analysis))
    blocks.append(_backend_block(run))
    blocks.append(_losses_block(run))
    return "\n".join(block for block in blocks if block)


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


def _header(run: MergedRun) -> str:
    """Runtime, process and worker counts, and what the sampler could measure."""
    runtime = max((w.written_at - w.started_at for w in run.workers), default=0.0)
    roles = ", ".join(f"{role} x{len(run.workers_of(role))}" for role in run.roles) or "none"
    return (
        f"Runtime {format_ns(runtime * 1e9)}   "
        f"Processes {len({w.pid for w in run.workers})}   "
        f"Roles {roles}\n"
        f"Host {run.metadata.get('host', '?')}"
    )


def _role_block(run: MergedRun, role: str) -> str:
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
        _RULE,
    ]
    lines.extend(_share_rows(tree))
    lines.append("")
    lines.extend(_dominant_rows(tree))
    lines.extend(_iteration_rows(tree))
    return "\n".join(lines)


def _share_rows(tree: PhaseTree) -> list[str]:
    """The pipeline breakdown: sibling phases as a share of their parent's wall time.

    A loop wrapped in a single outer phase would otherwise render as one 100% row, which
    says nothing. So the breakdown descends past any level that has only one phase, until
    it finds the first real split — that is the level where the work divides.
    """
    prefix = _first_branching_prefix(tree)
    siblings = [p for p in tree if len(p) == len(prefix) + 1 and p[:-1] == prefix]
    parent = tree.get(prefix)
    unattributed = parent.self_ns if parent is not None else 0
    total = sum(tree[path].wall_ns for path in siblings) + unattributed
    if total <= 0:
        return []

    rows = []
    for path in sorted(siblings, key=lambda p: -tree[p].wall_ns):
        stats = tree[path]
        rows.append(f"{path[-1]:<28}{100.0 * stats.wall_ns / total:>7.1f}%"
                    f"{format_ns(stats.wall_ns):>14}")
    if unattributed > total * 0.001:
        share = 100.0 * unattributed / total
        rows.append(f"{'Other':<28}{share:>7.1f}%{format_ns(unattributed):>14}")
    return rows


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
    """
    ranked = sorted(tree.items(), key=lambda item: -item[1].self_ns)
    rows = [f"{'DOMINANT PHASES':<28}{'self':>12}{'wait':>8}{'p50':>10}{'p99':>10}"]
    for path, stats in ranked[:limit]:
        if not path or stats.self_ns <= 0:
            continue
        wait_share = 100.0 * stats.wait_ns / stats.wall_ns if stats.wall_ns else 0.0
        rows.append(
            f"{_label(path):<28}{format_ns(stats.self_ns):>12}{wait_share:>7.0f}%"
            f"{format_ns(stats.hist.quantile(0.5)):>10}"
            f"{format_ns(stats.hist.quantile(0.99)):>10}",
        )
        rows.extend(_counter_rows(stats.counters, stats.wall_ns))
    return rows if len(rows) > 1 else []


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


def _counter_rows(counters: dict[str, int], wall_ns: int) -> list[str]:
    """Work counters and their rate per second of the phase's wall time.

    The ``io_*`` counters are skipped: they hold bytes, not work units, so a "per each"
    figure would be nonsense. They are rendered by :func:`_exact_io_block` instead.
    """
    seconds = wall_ns / 1e9
    rows = []
    for name, total in sorted(counters.items()):
        if name.startswith("io_"):
            continue
        rate = total / seconds if seconds else 0.0
        per_unit = wall_ns / total if total else 0.0
        rows.append(f"    + {name:<22}{total:>10,}{rate:>12,.1f}/s{format_ns(per_unit):>10}/ea")
    return rows


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
        f"  {'/'.join(path)[-26:]:<26}"
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
    if not (totals.read_bytes or totals.write_bytes or totals.read_chars or totals.write_chars):
        return ""

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
            f"  {phase[-26:]:<26}"
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
            f"  {reads:.0%} of reads and {writes:.0%} of writes moved while no phase was",
        )
        lines.append("   open — too coarse to attribute. Wrap those regions in io=True.")
    return lines


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
    if analysis.gpu_util_mean < 0 and not analysis.gpu_devices and not analysis.peak_cuda_reserved:
        return ""
    lines = ["", "GPU", _RULE]
    lines.extend(_gpu_utilisation_rows(analysis))
    if analysis.peak_cuda_reserved:
        lines.append(f"{'VRAM allocated (peak)':<28}{format_bytes(analysis.peak_cuda_alloc):>14}")
        lines.append(f"{'VRAM reserved (peak)':<28}{format_bytes(analysis.peak_cuda_reserved):>14}")
    lines.append("")
    lines.extend(_gpu_footnote(analysis))
    return "\n".join(lines)


def _gpu_footnote(analysis: SampleAnalysis) -> list[str]:
    """Say what the numbers above are, and what they are not."""
    if analysis.gpu_devices:
        return [
            "  (busy is NVML's whole-device percentage — every process's kernels, not",
            "   just yours; 'this run' is the share NVML attributes to this run's own",
            "   pids. Neither is a compute-vs-wait split: for that, run with",
            "   backend='torch' and analyse the trace.)",
        ]
    return [
        "  (utilisation is whole-device busy time from NVML, not a compute-vs-wait",
        "   split. For that, run with backend='torch' and analyse the trace.)",
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
        rows.append(f"  growing under {phase[-24:]:<24}{rate:>14}")
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
    """Name any worker whose data could not be read, rather than quietly under-reporting."""
    if not run.unreadable:
        return ""
    lines = ["", f"LOST: {len(run.unreadable)} worker file(s) unreadable", _RULE]
    lines.extend(f"  {path.name}" for path in run.unreadable)
    return "\n".join(lines)


def _label(path: PhasePath) -> str:
    """Render a phase path compactly: the leaf, with its parent when that disambiguates."""
    if len(path) == 1:
        return path[0]
    return f"{path[-2]}/{path[-1]}"[:27]
