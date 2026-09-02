"""HTML rendering of a merged run: an icicle chart of the phase tree, plus the sampled blocks.

The text report answers "where did the time go?" in a form you can read in a terminal or diff
in CI. This one answers the same question in a form you can attach to a ticket, and adds the
one thing a column of numbers genuinely cannot show: the *shape* of the phase tree, and which
parts of it were waiting rather than working.

It draws no conclusions the text report does not. Both consume the same derivations from
``report.py`` — ``sibling_shares`` and ``wait_share`` — precisely so the two can never
disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineprofiler.accounting.analysis import SampleAnalysis, analyse_processes, format_bytes
from lineprofiler.accounting.hardware import format_gpu_models
from lineprofiler.accounting.hardware import total_vram as hardware_total_vram
from lineprofiler.accounting.phasetree import PhasePath, PhaseTree
from lineprofiler.accounting.provenance import source_of
from lineprofiler.accounting.report import (
    format_ns,
    idle_context_roles,
    pooled_capacity,
    report_as_dict,
    sibling_shares,
    wait_share,
)
from lineprofiler.accounting.snapshot import MergedRun
from lineprofiler.htmldoc import clip_label, document, escape, tile

# A cell thinner than this is narrower than its own border at any sane figure width, so it
# would read as a smudge rather than a phase. They are counted and reported, never dropped
# silently — an invisible phase and an absent one must not look the same.
_MIN_VISIBLE_WIDTH = 0.002

_ROW_HEIGHT = 22
_CHART_WIDTH = 1000


@dataclass(frozen=True, slots=True)
class FlameCell:
    """One rectangle of the icicle chart: a phase, its depth, and its horizontal span.

    ``x`` and ``width`` are fractions of the full chart width, so the caller picks the pixel
    size. ``wait_pct`` drives the colour, which is the whole reason to draw this rather than
    print the same numbers in a table.
    """

    path: PhasePath
    depth: int
    x: float
    width: float
    wall_ns: int
    self_ns: int
    calls: int
    wait_pct: float

    @property
    def label(self) -> str:
        """The phase's own name, without its ancestors."""
        return self.path[-1] if self.path else ""


def flame_cells(tree: PhaseTree) -> list[FlameCell]:
    """Lay a phase tree out as an icicle chart, widest child leftmost.

    Children are scaled to fit inside their parent rather than against a global total. A
    parent whose children's wall times sum to more than its own — which happens honestly,
    when a phase is entered recursively or from two threads — would otherwise draw children
    overflowing their parent, which reads as a measurement error rather than as the
    aggregation it is.
    """
    roots = [path for path in tree if len(path) == 1]
    if not roots:
        return []

    total = sum(tree[path].wall_ns for path in roots)
    if total <= 0:
        return []

    cells: list[FlameCell] = []
    _place_children(tree, (), 0.0, 1.0, 0, cells, total)
    return cells


def _place_children(
    tree: PhaseTree,
    parent: PhasePath,
    x: float,
    width: float,
    depth: int,
    cells: list[FlameCell],
    span_ns: int,
) -> None:
    """Place ``parent``'s children across ``width``, then recurse into each."""
    children = sorted(
        (p for p in tree if len(p) == len(parent) + 1 and p[:-1] == parent),
        key=lambda p: -tree[p].wall_ns,
    )
    if not children or span_ns <= 0 or width <= 0:
        return

    # The guard described in flame_cells: never let children exceed the space they sit in.
    occupied = max(span_ns, sum(tree[path].wall_ns for path in children))
    cursor = x
    for path in children:
        stats = tree[path]
        child_width = width * stats.wall_ns / occupied
        cells.append(
            FlameCell(
                path=path,
                depth=depth,
                x=cursor,
                width=child_width,
                wall_ns=stats.wall_ns,
                self_ns=stats.self_ns,
                calls=stats.calls,
                wait_pct=wait_share(stats),
            ),
        )
        _place_children(tree, path, cursor, child_width, depth + 1, cells, stats.wall_ns)
        cursor += child_width


def render_html(run: MergedRun, title: str = "lineprofiler report") -> str:
    """Return a complete, self-contained HTML document for a merged run.

    The ``report_as_dict`` document is embedded alongside the figures — not used to draw
    them — so the page carries the exact numbers a script would read, and a reader can check
    any figure against them without re-running the merge.
    """
    analysis = analyse_processes(run.samples_by_process())
    blocks = [
        _header(run, title),
        _resources_block(analysis, run),
        *(_role_block(run, role) for role in run.roles),
        _io_block(analysis),
        _gpu_block(analysis),
        _memory_block(analysis),
        _caveats_block(run),
    ]
    body = "\n".join(block for block in blocks if block)
    return document(title, body, data=report_as_dict(run))


def write_html(run: MergedRun, path: str | Path, title: str = "lineprofiler report") -> None:
    """Write :func:`render_html` to ``path``, creating parent directories if needed.

    Unlike the CLI, which lets a mistyped path fail loudly, the library call creates the
    directory: a caller writing to ``reports/run-17.html`` from code means it.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(run, title), encoding="utf-8")


def _header(run: MergedRun, title: str) -> str:
    runtime = max((w.written_at - w.started_at for w in run.workers), default=0.0)
    roles = ", ".join(f"{role} x{len(run.workers_of(role))}" for role in run.roles)
    tiles = "".join([
        tile("runtime", format_ns(runtime * 1e9)),
        tile("processes", str(len(run.workers))),
        tile("roles", roles or "none"),
        tile("hosts", ", ".join(run.hosts) or "unknown"),
        tile("imbalance", f"{run.imbalance:.2f}"),
    ])
    run_id = escape(str(run.metadata.get("run_id", "unknown")))
    source = source_of(run.metadata)
    source_html = f" · {escape(source)}" if source else ""
    return (
        f"<h1>{escape(title)}</h1>\n"
        f'<p class="sub mono">run {run_id}{source_html}</p>\n'
        f'<div class="tiles">{tiles}</div>'
    )


def _resources_block(analysis: SampleAnalysis, run: MergedRun) -> str:
    """Used-versus-available headlines, plus one inventory row per machine.

    The table is the part a reader compares between two servers, and the only place the device
    models appear. Both it and the tiles omit what was not recorded rather than showing zero.
    """
    tiles = _resource_tiles(analysis, run)
    inventory = _inventory_table(run)
    if not tiles and not inventory:
        return ""
    parts = ["<h2>Resources</h2>"]
    if tiles:
        parts.append(f'<div class="tiles">{tiles}</div>')
    # Beside the tiles, not in a trailing block: the pair "peak VRAM 290.7 MB" and "VRAM held
    # 2.0 GB" invites the question here, and an explanation eighty lines below is one a reader
    # who stops at the headline figures never reaches.
    parts.extend(_vram_instrument_note(analysis))
    parts.extend(_idle_context_notes(run))
    if inventory:
        parts.append(inventory)
    return "\n".join(parts)


def _vram_instrument_note(analysis: SampleAnalysis) -> list[str]:
    """Say which instrument each of the two VRAM tiles came from."""
    if not analysis.cuda_process_measured or not analysis.peak_cuda_alloc:
        return []
    return [
        '<p class="note">“peak VRAM” is the torch allocator (tensors only); “VRAM held” is '
        "what the device reports for this run's pids, which includes each process's CUDA "
        "context — typically a few hundred MB per process the allocator never sees.</p>",
    ]


def _idle_context_notes(run: MergedRun) -> list[str]:
    """Name any role holding VRAM it never allocated a tensor in, and what to do about it."""
    return [
        f'<p class="note"><strong>{escape(str(processes))} {escape(role)} process(es) hold '
        f"{escape(format_bytes(each))} of VRAM each with no allocator activity</strong> — a "
        "CUDA context in a process doing no GPU work. Pass <code>cuda_sync=False</code> to "
        "that role's Profiler if it holds no model.</p>"
        for role, processes, each in idle_context_roles(run)
    ]


def _resource_tiles(analysis: SampleAnalysis, run: MergedRun) -> str:
    """Consumption headlines, each paired with capacity where the run recorded it."""
    capacity = pooled_capacity(run)
    processes = max(len(run.workers), 1)
    tiles: list[str] = []
    if analysis.cpu.measured:
        cores = capacity.get("cpu_affinity") or capacity.get("cpu_cores")
        peak = f"{analysis.cpu.peak:.1f}"
        tiles.append(tile("peak CPU", f"{peak} / {cores} cores" if cores else f"{peak} cores"))
        tiles.append(tile("CPU per process", f"{analysis.cpu.peak / processes:.2f} cores"))
    if analysis.memory.peak_rss:
        used = format_bytes(analysis.memory.peak_rss)
        total = capacity.get("ram_total")
        tiles.append(tile("peak RSS", f"{used} / {format_bytes(total)}" if total else used))
        tiles.append(tile("RSS per process", format_bytes(analysis.memory.peak_rss / processes)))
    vram = hardware_total_vram(capacity.get("gpus", []))
    if analysis.peak_cuda_alloc:
        used = format_bytes(analysis.peak_cuda_alloc)
        tiles.append(tile("peak VRAM", f"{used} / {format_bytes(vram)}" if vram else used))
    # Against the same denominator, and the only one of the two that belongs there: the
    # allocator's peak omits every process's CUDA context, so putting it over a device total
    # answers "can I add a worker?" with a number that cannot see what a worker costs.
    if analysis.cuda_process_measured:
        held = format_bytes(analysis.peak_cuda_process)
        tiles.append(tile("VRAM held", f"{held} / {format_bytes(vram)}" if vram else held))
    if tiles:
        tiles.append(tile("processes", str(processes)))
    return "".join(tiles)


def _inventory_table(run: MergedRun) -> str:
    """One row per host: cores, the job's share of them, RAM and devices."""
    by_host = run.hardware_by_host
    if not by_host:
        return ""
    rows = "".join(_inventory_row(host, hardware) for host, hardware in by_host.items())
    return (
        '<div class="scroll"><table><thead><tr><th>host</th><th>cores</th>'
        "<th>available</th><th>RAM</th><th>GPUs</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _inventory_row(host: str, hardware: dict[str, Any]) -> str:
    """One machine. Every cell is escaped: device names reach this page from the driver."""
    cores = hardware.get("cpu_cores")
    affinity = hardware.get("cpu_affinity")
    ram = hardware.get("ram_total")
    gpus = hardware.get("gpus") or []
    cells = [
        escape(host),
        escape(str(cores)) if cores else "—",
        escape(str(affinity)) if affinity else "—",
        escape(format_bytes(ram)) if ram else "—",
        escape(format_gpu_models(gpus)) if gpus else "—",
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _role_block(run: MergedRun, role: str) -> str:
    """One role's pipeline breakdown and icicle chart."""
    tree = run.tree_of(role)
    shares = sibling_shares(tree)
    if not shares:
        return ""

    rows = "".join(
        f"<tr><td>{escape(clip_label(share.name))}</td><td>{share.percent:.1f}%</td>"
        f"<td>{escape(format_ns(share.wall_ns))}</td></tr>"
        for share in shares
    )
    table = (
        '<div class="scroll"><table><thead><tr><th>phase</th><th>share</th>'
        "<th>wall</th></tr></thead><tbody>"
        f"{rows}</tbody></table></div>"
    )
    return (
        f"<h2>{escape(role)} — {len(run.workers_of(role))} process(es)</h2>\n"
        f"{table}\n{_flamegraph(tree)}"
    )


def _flamegraph(tree: PhaseTree) -> str:
    """The icicle chart, as inline SVG with native tooltips and no script."""
    cells = flame_cells(tree)
    if not cells:
        return ""

    drawn = [cell for cell in cells if cell.width >= _MIN_VISIBLE_WIDTH]
    height = (max(cell.depth for cell in drawn) + 1) * _ROW_HEIGHT if drawn else 0
    rects = "".join(_flame_rect(cell) for cell in drawn)
    hidden = len(cells) - len(drawn)
    note = (
        f'<p class="note">{hidden} phase(s) too narrow to draw; '
        f"they are in the data block and the text report.</p>"
        if hidden
        else ""
    )
    return (
        f'<div class="scroll"><svg viewBox="0 0 {_CHART_WIDTH} {height}" '
        f'width="{_CHART_WIDTH}" height="{height}" role="img" '
        f'aria-label="phase tree, width is wall time">{rects}</svg></div>{note}\n'
        f'<p class="note">Width is wall time; colour is the share of it spent waiting '
        f"rather than running.</p>"
    )


def _flame_rect(cell: FlameCell) -> str:
    x = cell.x * _CHART_WIDTH
    width = max(cell.width * _CHART_WIDTH, 1.0)
    y = cell.depth * _ROW_HEIGHT
    fill = _wait_colour(cell.wait_pct)
    tooltip = escape(
        f"{clip_label('/'.join(cell.path))}\n"
        f"wall {format_ns(cell.wall_ns)}  self {format_ns(cell.self_ns)}\n"
        f"calls {cell.calls}  wait {cell.wait_pct:.0f}%",
    )
    # Roughly seven pixels a character; a label that cannot fit is dropped rather than
    # allowed to spill across its neighbours.
    label = ""
    if width > len(cell.label) * 7 + 8:
        label = (
            f'<text x="{x + 4:.1f}" y="{y + 15}" font-size="11" '
            f'fill="#ffffff">{escape(cell.label)}</text>'
        )
    return (
        f'<g><title>{tooltip}</title>'
        f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="{_ROW_HEIGHT - 2}" '
        f'fill="{fill}" stroke="#ffffff" stroke-width="1"/>{label}</g>'
    )


def _wait_colour(wait_pct: float) -> str:
    """Blend from the working colour to the waiting one across 0–100% wait.

    A phase that is mostly blocked is usually the answer in a queue-driven pipeline, and it
    is invisible in a chart that colours by name or by depth.
    """
    fraction = min(max(wait_pct / 100.0, 0.0), 1.0)
    working = (0xB2, 0x3C, 0x17)
    waiting = (0x2F, 0x6F, 0x9F)
    blended = tuple(
        round(start + (end - start) * fraction)
        for start, end in zip(working, waiting, strict=True)
    )
    return "#{:02x}{:02x}{:02x}".format(*blended)


def _io_block(analysis: SampleAnalysis) -> str:
    if not analysis.has_samples:
        return ""
    totals = analysis.totals
    tiles = "".join([
        tile("read (disk)", format_bytes(totals.read_bytes)),
        tile("read (cache)", format_bytes(totals.cached_read_bytes)),
        tile("written", format_bytes(totals.write_bytes)),
        tile("read rate", f"{format_bytes(int(totals.read_rate))}/s"),
    ])
    gap = (
        f'<p class="note">{analysis.io_gap_intervals} sample interval(s) could not be read, '
        f"so these totals are a floor rather than a measurement.</p>"
        if analysis.io_gap_intervals
        else ""
    )
    return (
        f"<h2>I/O</h2>\n<div class=\"tiles\">{tiles}</div>\n"
        f"{_series_chart(analysis.read_series, 'read')}"
        f"{_series_chart(analysis.write_series, 'write')}{gap}"
    )


def _series_chart(series: list[float], label: str) -> str:
    """One rate series as a filled polyline. No axes: the shape is the message, not the scale."""
    if not series or max(series) <= 0:
        return ""
    peak = max(series)
    width, height = 1000, 60
    step = width / max(len(series) - 1, 1)
    points = " ".join(
        f"{index * step:.1f},{height - (value / peak) * height:.1f}"
        for index, value in enumerate(series)
    )
    return (
        f'<p class="note">{escape(label)} — peak {format_bytes(int(peak))}/s</p>'
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{escape(label)} rate over the run">'
        f'<polyline points="{points}" fill="none" stroke="var(--cool)" '
        f'stroke-width="1.5"/></svg></div>'
    )


def _gpu_block(analysis: SampleAnalysis) -> str:
    # Shares `has_gpu` with the text report so the two can never disagree about whether this
    # run had a GPU in it. The device table itself still needs per-device rows to draw.
    if not analysis.has_gpu:
        return ""
    if not analysis.gpu_devices:
        return (
            f"<h2>GPU</h2>\n<div class=\"tiles\">{_vram_tiles(analysis)}</div>\n"
            '<p class="note">No per-device utilisation was recorded — install nvidia-ml-py '
            "to collect it.</p>"
            f"{_held_vram_note(analysis)}"
        )
    rows = "".join(
        f"<tr><td>GPU {device.index}</td><td>{device.busy_mean:.1f}%</td>"
        f"<td>{'n/a' if device.ours_mean < 0 else f'{device.ours_mean:.1f}%'}</td></tr>"
        for device in analysis.gpu_devices
    )
    return (
        f"<h2>GPU</h2>\n<div class=\"tiles\">{_vram_tiles(analysis)}</div>\n"
        '<div class="scroll"><table><thead><tr><th>device</th><th>busy</th>'
        "<th>this run</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>\n"
        '<p class="note">“busy” is the whole device including other tenants; '
        "“this run” is what NVML attributed to these processes.</p>"
        f"{_held_vram_note(analysis)}"
    )


def _vram_tiles(analysis: SampleAnalysis) -> str:
    """The allocator's two figures, and what the device says this run actually holds."""
    tiles = [
        tile("VRAM allocated", format_bytes(analysis.peak_cuda_alloc)),
        tile("VRAM reserved", format_bytes(analysis.peak_cuda_reserved)),
    ]
    if analysis.cuda_process_measured:
        tiles.append(tile("VRAM held", format_bytes(analysis.peak_cuda_process)))
    return "".join(tiles)


def _held_vram_note(analysis: SampleAnalysis) -> str:
    """Say which instrument each VRAM tile came from, next to the tiles themselves.

    Without this the two read as a rounding disagreement rather than as two different
    measurements, and the gap between them — the per-process CUDA context — is the term that
    decides whether another worker fits on the card.
    """
    if not analysis.cuda_process_measured:
        return ""
    return (
        '\n<p class="note">“allocated” and “reserved” are the torch caching allocator '
        "(tensors only); “held” is what the device reports for this run's pids, which "
        "includes each process's CUDA context — typically a few hundred MB per process that "
        "the allocator never sees.</p>"
    )


def _memory_block(analysis: SampleAnalysis) -> str:
    if not analysis.has_samples:
        return ""
    memory = analysis.memory
    tiles = "".join([
        tile("peak RSS", format_bytes(memory.peak_rss)),
        tile("final RSS", format_bytes(memory.last_rss)),
        tile("growth", format_bytes(memory.growth_bytes)),
        tile("slope", f"{format_bytes(int(memory.slope_bytes_per_s))}/s"),
    ])
    return f'<h2>Memory</h2>\n<div class="tiles">{tiles}</div>'


def _caveats_block(run: MergedRun) -> str:
    """Anything that makes the run less than complete, in the page as well as the data.

    A run that lost a worker must not read as a whole result here either.
    """
    items = []
    if run.unreadable:
        items.append(f"{len(run.unreadable)} worker file(s) could not be read")
    if run.superseded:
        items.append(f"{len(run.superseded)} worker file(s) from a superseded attempt")
    if not items:
        return ""
    entries = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<h2>Caveats</h2>\n<ul>{entries}</ul>"
