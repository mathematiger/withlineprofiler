"""The trace timeline: lanes on one clock, wait shading, arrows, and the critical path.

The other two pages answer "where did the time go?". This one answers the question a set of
totals structurally cannot: *why was this worker idle at that moment, and who was it waiting
for?* A phase tree can say ``queue_get`` was 80% wait; only a timeline can show that the wait
began the instant the learner finished and ended when actor 3 finally published its batch.

This is the one page in the package that ships JavaScript. The constraint it relaxes is
deliberate and narrow: still one file, still no CDN, no webfont and no network — but a
timeline over a hundred thousand spans needs pan and zoom, and static SVG cannot provide
them. The report and source pages remain script-free, and their tests still assert it.

Drawing decisions worth knowing:

- **Colour means wait, never identity.** The same blend the icicle chart uses, so the two
  pages agree: reddish is working, blue is blocked. A span whose CPU time was not measured is
  drawn hatched grey rather than as one that never waited.
- **The critical path is outlined, not recoloured.** It is an overlay on the answer, not a
  replacement for it.
- **Idle time is drawn as absence**, so a lane full of gaps reads as a lane full of gaps.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from lineprofiler.accounting.analysis import analyse_processes
from lineprofiler.accounting.findings import (
    Finding,
    PhaseTotal,
    phase_totals,
    rank_findings,
)
from lineprofiler.accounting.provenance import source_of
from lineprofiler.accounting.report import format_ns
from lineprofiler.accounting.snapshot import MergedRun
from lineprofiler.accounting.trace import FLAG_AUTO, Origin
from lineprofiler.accounting.tracealign import (
    AlignedTrace,
    PlacedSpan,
    align_run,
    alignment_accuracy_note,
    clock_step_note,
    critical_path,
    lane_busy_share,
    lane_working_share,
    max_depth_of,
)
from lineprofiler.htmldoc import (
    MAX_LABEL_DRAWN,
    JsonValue,
    clip_label,
    document,
    escape,
    tile,
)

_MAX_SPANS_DRAWN = 120_000
"""Spans handed to the page. Past this the canvas stays smooth but the file gets large, so
the longest spans are kept and the count of omitted ones is stated. Never a silent trim."""

_MAX_DEPTH_DRAWN = 8
"""Nesting levels given their own row before deeper spans are folded onto the last one.

Phases nest to 32 and auto-derived spans to 64, so an unbounded row per level would let one
deeply recursive lane set the height of the whole chart. Folding rather than dropping keeps
the span visible; the count of folded spans is reported in the caveats, never hidden."""

_MAX_SEQUENCE_ROWS = 40
"""Calls listed per lane in the sequence table. The chart holds the rest; the table is there
to be read in order, and a few hundred rows of it would not be."""

_MAX_SUMMARY_ROWS = 25
"""Phases listed in the summary. Enough to cover what a run is made of; past it the table
stops being a ranking and becomes a dump, and the embedded data block still carries them all."""


def render_trace_html(
    run: MergedRun,
    title: str = "lineprofiler trace",
    max_spans: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Return a complete, self-contained timeline page for a merged run.

    Raises nothing when a run has no trace data: the page says so plainly and explains how to
    record some, which is more useful than an exception to someone who has just discovered
    the feature exists.

    ``max_spans`` overrides the built-in cap, so a run too large to render at the default
    degrades to a readable picture rather than failing after the expensive half of the work
    has already succeeded. ``progress`` receives coarse status lines; the stages it reports
    are the slow ones, so a caller can tell a long render from a stuck one.
    """
    say = progress if progress is not None else _silent
    aligned = align_run(run)
    say(f"aligned {len(aligned.spans):,} spans across {len(aligned.lanes)} lanes")
    if not aligned.spans:
        return document(title, _empty_body(title), data={"spans": []})

    _warn_if_large(aligned, max_spans, say)
    chain = critical_path(aligned)
    say(f"critical path: {len(chain)} spans")
    findings = rank_findings(aligned)
    totals = phase_totals(aligned)
    say(f"{len(findings)} finding(s) across {len(totals)} distinct phases")
    payload = _payload(aligned, chain, run, max_spans)
    omitted = payload["omitted"]
    if isinstance(omitted, int) and omitted:
        say(f"omitted {omitted:,} spans below the drawing cap")
    # Conclusions first, evidence after. Someone opening this wants to know what is wrong
    # before they are asked to read a chart; the timeline is where they go to confirm it,
    # which is the opposite of the order the page used to impose on them.
    body = "\n".join([
        _header(aligned, title, run),
        _findings_block(findings, aligned.hosts),
        _phase_summary(totals, aligned),
        _canvas_block(_gpu_device_count(payload)),
        _critical_path_block(chain, aligned),
        _lane_summary(aligned),
        _sequence_block(aligned),
        _caveats(aligned, omitted if isinstance(omitted, int) else 0),
    ])
    say("writing HTML")
    return document(title, body, data=payload, style=_STYLE, script=_SCRIPT)


def _silent(message: str) -> None:
    """Default progress sink: the library call stays quiet unless asked not to."""


def _warn_if_large(
    aligned: AlignedTrace,
    max_spans: int | None,
    say: Callable[[str], None],
) -> None:
    """Say up front that this will be slow, and name the flag that makes it fast.

    Estimated before the work rather than discovered during it: the point is to let someone
    interrupt a render they did not want, which is only useful while it is still starting.
    """
    limit = max_spans if max_spans is not None else _MAX_SPANS_DRAWN
    if len(aligned.spans) <= limit:
        return
    # Only suggest the flag to someone who has not already used it; repeating the advice back
    # at a caller who took it reads as the tool not knowing what it was asked to do.
    advice = "" if max_spans is not None else " — pass --max-spans to draw fewer"
    say(f"{len(aligned.spans):,} spans exceeds the {limit:,} drawn; this may take a while{advice}")


def write_trace_html(
    run: MergedRun,
    path: str | Path,
    title: str = "lineprofiler trace",
    max_spans: int | None = None,
) -> None:
    """Write :func:`render_trace_html` to ``path``, creating parent directories if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_trace_html(run, title, max_spans), encoding="utf-8")


def _empty_body(title: str) -> str:
    """What to show when the run carries no spans, rather than an empty chart."""
    return (
        f"<h1>{escape(title)}</h1>\n"
        '<p class="sub">This run has no trace data, so there is no timeline to draw.</p>\n'
        "<h2>Recording a trace</h2>\n"
        "<p>Tracing is off by default because a timeline is unbounded where the phase tree "
        "is not. Turn it on at the profiler, or in the environment:</p>\n"
        '<pre class="mono">Profiler(run_dir="profile", role="actor", trace=True)\n\n'
        "# or, with no code change at all:\n"
        "export LINEPROFILER_TRACE=1        # your named phases\n"
        "export LINEPROFILER_TRACE=auto     # derived from function calls</pre>\n"
        '<p class="note">Then re-run, and read the timeline with '
        '<span class="mono">lineprofiler trace &lt;run_dir&gt; -o trace.html</span>.</p>'
    )


def _header(aligned: AlignedTrace, title: str, run: MergedRun) -> str:
    """Headline figures: how long, how many lanes, and how much of it was waiting.

    A rejected clock anchor is disclosed *here*, not only in the caveats: it qualifies the
    axis every figure below is measured along, and this page has already learned once that a
    caveat eighty lines under a confident conclusion does not reach whoever acts on it.
    """
    waiting = _overall_wait_share(aligned)
    tiles = "".join([
        # "traced span", not "runtime": this is first-span to last-span, which is legitimately
        # shorter than the run's wall clock — a worker profiles for a while before its first
        # phase and after its last. The two figures differing is expected, and an unlabelled
        # difference reads as a broken timebase.
        tile("traced span", format_ns(aligned.duration_ns)),
        tile("lanes", str(len(aligned.lanes))),
        tile("spans", f"{len(aligned.spans):,}"),
        tile("arrows", str(len(aligned.arrows))),
        tile("waiting", "n/a" if waiting < 0 else f"{waiting:.0f}%"),
    ])
    run_id = escape(str(run.metadata.get("run_id", "unknown")))
    source = source_of(run.metadata)
    source_html = f" · {escape(source)}" if source else ""
    stepped = clock_step_note(aligned.clock_steps)
    stepped_html = f'<p class="warn">{escape(stepped)}</p>\n' if stepped else ""
    return (
        f"<h1>{escape(title)}</h1>\n"
        f'<p class="sub mono">run {run_id}{source_html}</p>\n'
        f'<div class="tiles">{tiles}</div>\n'
        f"{stepped_html}"
        '<p class="note">“traced span” is first span to last, which is shorter than the '
        "run's wall clock: a worker starts before its first phase and ends after its last.</p>"
    )


def _overall_wait_share(aligned: AlignedTrace) -> float:
    """Share of measured span time spent blocked, ``-1.0`` when nothing measured CPU."""
    measured = [span for span in aligned.spans if span.cpu_measured]
    if not measured:
        return -1.0
    total = sum(span.duration_ns for span in measured)
    if total <= 0:
        return 0.0
    return 100.0 * sum(span.wait_ns for span in measured) / total


def _lane_summary(aligned: AlignedTrace) -> str:
    """Per-lane occupancy, which is where an idle worker becomes obvious as a number.

    Two columns rather than one, because they answer different questions: "phase open" counts
    a lane blocked inside ``queue_get`` as occupied, "on CPU" does not. A large gap between
    them is precisely the symptom this page exists to explain.
    """
    rows = []
    for lane in aligned.lanes:
        busy = lane_busy_share(aligned, lane)
        working = lane_working_share(aligned, lane)
        role = aligned.roles.get(lane, "")
        working_cell = "n/a" if working < 0 else f"{working:.1f}%"
        idle = "n/a" if working < 0 else f"{max(0.0, busy - working):.1f}%"
        rows.append(
            f"<tr><td>{escape(lane)}</td><td>{escape(role)}</td>"
            f"<td>{busy:.1f}%</td><td>{working_cell}</td><td>{idle}</td></tr>",
        )
    return (
        "<h2>Lanes</h2>\n"
        '<div class="scroll"><table><thead><tr><th>lane</th><th>role</th>'
        "<th>phase open</th><th>on CPU</th><th>blocked</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>\n"
        '<p class="note">“phase open” counts a lane blocked in a queue as occupied; '
        "“on CPU” does not. The difference is the waiting.</p>"
    )


def _sequence_block(aligned: AlignedTrace) -> str:
    """What each lane called, in the order it called it — the chart's claim as a list.

    The timeline shows order by position, which is the right way to see it and the wrong way
    to quote it: a dense lane collapses into a stripe, and a nested call is a bar under a bar.
    This says the same thing exactly, so "the learner runs queue_get then train_step, every
    iteration" can be read off rather than inferred from pixels.
    """
    if not aligned.spans:
        return ""
    origin = aligned.t0_ns
    sections = []
    for lane in aligned.lanes:
        spans = sorted(
            (span for span in aligned.spans if span.lane == lane),
            key=lambda span: (span.t0_ns, span.depth),
        )
        rows = "".join(_sequence_row(span, origin) for span in spans[:_MAX_SEQUENCE_ROWS])
        omitted = len(spans) - _MAX_SEQUENCE_ROWS
        more = (
            f'<p class="note">{omitted:,} further call(s) on this lane are not listed; '
            "the timeline above shows all of them.</p>"
            if omitted > 0
            else ""
        )
        role = aligned.roles.get(lane, "")
        # Built as its own statement rather than nested inside the f-string below: a
        # backslash in an f-string expression is a SyntaxError before 3.12, and the package
        # supports 3.10.
        role_span = f' <span class="note">{escape(role)}</span>' if role else ""
        sections.append(
            f"<h3>{escape(lane)}{role_span}</h3>\n"
            '<div class="scroll"><table><thead><tr><th>#</th><th>call</th><th>at</th>'
            "<th>duration</th><th>wait</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>{more}",
        )
    return (
        "<h2>Call order</h2>\n"
        '<p class="note">Each lane in the order it ran, indentation showing what was called '
        "from inside what. Read down for the sequence, right for the nesting.</p>\n"
        + "\n".join(sections)
    )


def _cell_label(text: str) -> str:
    """A phase name as a table cell: bounded, escaped, and honest about the cut.

    The tail is kept for the same reason :func:`report.format_label` keeps it — the leaf is
    what a reader is looking for — and the cut is marked, because an unmarked truncation
    prints a name that does not exist. The ``title`` says how much was dropped rather than
    carrying the rest: a tooltip holding the whole name would put it back into the markup
    once per table, which is the cost this bound exists to avoid. The complete name is in the
    embedded JSON, which is where a reader goes for exact values anyway.
    """
    cut = clip_label(text)
    if cut == text:
        return escape(text)
    hidden = len(text) - (MAX_LABEL_DRAWN - 1)
    return (
        f'<span title="name truncated for display; {hidden:,} leading characters not shown '
        f'— the full name is in the data block">{escape(cut)}</span>'
    )


def _sequence_row(span: PlacedSpan, origin: int) -> str:
    """One call in a lane's sequence, indented to show what it was called from."""
    indent = "&nbsp;" * (4 * min(span.depth, _MAX_DEPTH_DRAWN))
    arrow = "↳ " if span.depth else ""
    wait = "n/a" if span.wait_pct < 0 else f"{span.wait_pct:.0f}%"
    return (
        f'<tr><td class="mono">{span.depth}</td>'
        f'<td class="mono">{indent}{arrow}{_cell_label(span.name)}</td>'
        f"<td>{escape(format_ns(span.t0_ns - origin))}</td>"
        f"<td>{escape(format_ns(span.duration_ns))}</td>"
        f"<td>{wait}</td></tr>"
    )


def _findings_block(findings: list[Finding], hosts: set[str] | None = None) -> str:
    """What is wrong with this run, ranked, at the top of the page.

    The page used to open with a lane table and a canvas, which is evidence rather than a
    conclusion: it required the reader to already know what a healthy run looks like. This
    states the answer and lets the rest of the page justify it.

    Each finding that names something on the timeline gets a button rather than a sentence
    telling the reader to go and find it, so the connection between the claim and the picture
    survives the reader not knowing the vocabulary yet.

    On a run spanning hosts the block carries one line saying the axis is only as good as the
    clocks. Several of these findings — a lane idle while another worked, one lane active at a
    time, a phase blocked across lanes — are claims about the *relative* timing of processes,
    and across hosts that relation is NTP-accurate rather than exact. ``Caveats`` states this
    in full at the foot of the page, which is precisely where the superseded-worker disclosure
    used to sit while the header claimed one process: a reader who stops at the ranked
    conclusions, which is what this block is for, never gets there.
    """
    if not findings:
        return (
            "<h2>Findings</h2>\n"
            '<p class="note">No lane was idle for a large share of the run, no phase spent '
            "most of its time blocked, and no single lane dominated. There is no obvious "
            "bottleneck to name — the timeline below is still worth reading for the shape of "
            "the run.</p>"
        )
    items = "".join(_finding_item(index, finding) for index, finding in enumerate(findings))
    return (
        "<h2>Findings</h2>\n"
        '<p class="note">Ranked by how much of the traced span each one accounts for. '
        "“Show on timeline” filters the chart below to what the finding is about."
        f"{_findings_clock_note(hosts)}</p>\n"
        f'<ol class="findings">{items}</ol>'
    )


def _findings_clock_note(hosts: set[str] | None) -> str:
    """One sentence qualifying cross-host findings, or nothing on the ordinary one-host run.

    Deliberately short and deliberately not a repeat of :func:`alignment_accuracy_note`, which
    keeps the full statement in ``Caveats``. This is the pointer that stops a reader acting on
    a comparison between hosts without knowing it rests on NTP; most runs are one host, and a
    note they see every time is one they stop reading.
    """
    if not hosts or len(hosts) < 2:
        return ""
    return (
        f" These workers ran on {len(hosts)} hosts: any finding comparing one lane against "
        "another is only as accurate as the clocks agree — see Caveats."
    )


def _finding_item(index: int, finding: Finding) -> str:
    """One ranked finding: the claim, the evidence, and a way to see it."""
    action = ""
    if finding.is_actionable:
        # data-* rather than an inline handler: the page has a Content-Security-Policy-shaped
        # constraint of its own (no network, one file), and inline handlers would also mean
        # building markup out of a phase name, which is the XSS this page is careful about.
        action = (
            f'<button type="button" class="tl-jump" data-anchor="{escape(finding.anchor)}">'
            "show on timeline</button>"
        )
    return (
        f'<li class="finding" data-kind="{escape(finding.kind)}">'
        f'<div class="claim">{escape(finding.headline)}</div>'
        f'<div class="why">{escape(finding.detail)}</div>'
        f'<div class="act">{action}</div>'
        "</li>"
    )


def _phase_summary(totals: list[PhaseTotal], aligned: AlignedTrace) -> str:
    """Vampir's Function Summary: which phases the run is actually made of.

    The timeline answers *when*; this answers *how much*, which is the question a reader
    starts with and the one a timeline answers worst — a phase costing 40% of the run scattered
    over ten thousand short calls is invisible on a chart and top of this table.

    The bar is drawn in the same blend the chart uses, so a glance down the column separates
    "expensive because it works" from "expensive because it waits" without reading a number.
    """
    if not totals:
        return ""
    widest = max(total.wall_ns for total in totals) or 1
    # Shortened against what the listed rows share, so the column shows `envs/simulator.py`
    # rather than the machine-specific prefix every row repeats.
    root = _common_root(
        [total.origin.file for total in totals if total.origin is not None],
    )
    rows = "".join(
        _phase_row(total, widest, aligned, root) for total in totals[:_MAX_SUMMARY_ROWS]
    )
    omitted = len(totals) - _MAX_SUMMARY_ROWS
    more = (
        f'<p class="note">{omitted:,} further phase(s) are not listed; the embedded data '
        "block carries them all.</p>"
        if omitted > 0
        else ""
    )
    return (
        "<h2>Phase summary</h2>\n"
        '<p class="note">Every phase by total wall time across all lanes. “self” excludes '
        "time spent inside nested phases, so a wrapper does not out-rank the callee that "
        "spent the time. Bar colour is the share of the phase spent blocked. “wall” and "
        "“self” are summed over every lane the phase ran on, so they can exceed the run's "
        "duration; “share of lane time” divides by that same lane count, so it cannot. Rows "
        "are ordered by wall, which is why the share column does not fall strictly — a phase "
        "on two lanes outranks a longer one on a single lane.</p>\n"
        f"{_device_footnote(totals)}"
        '<div class="scroll"><table><thead><tr><th>phase</th><th>calls</th><th>lanes</th>'
        "<th>wall</th><th>self</th><th>blocked</th><th>share of lane time</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>{more}"
    )


def _device_footnote(totals: list[PhaseTotal]) -> str:
    """Define the ‡ mark, and only when a row on this page carries it."""
    if not any(total.on_device and total.wait_pct >= 0 for total in totals):
        return ""
    return (
        '<p class="note">‡ = the phase drained the CUDA queue at both ends '
        "(<code>sync=True</code>), so its “blocked” share is the device running that phase's "
        "own work — not a wait on another process.</p>\n"
    )


def _phase_row(
    total: PhaseTotal,
    widest: int,
    aligned: AlignedTrace,
    root: str = "",
) -> str:
    """One phase's totals, with a bar whose length is wall time and colour is its wait."""
    # Against the lane time this phase could have occupied, not the run's wall clock: a phase
    # running on two lanes has two lane-seconds available per wall second, so dividing a
    # summed wall time by the wall clock produced shares over 100% — which reads as a bug and
    # discredits every other figure on the page.
    available_ns = aligned.duration_ns * max(1, total.lanes)
    share = 100.0 * total.wall_ns / available_ns if available_ns else 0.0
    width = 100.0 * total.wall_ns / widest
    blocked = "n/a" if total.wait_pct < 0 else f"{total.wait_pct:.0f}%"
    # A synchronised phase's off-CPU time is the device running its own work, so the column
    # heading means the opposite of what it says on that row. Marked here rather than only in
    # the findings above: the table is what a reader ranks by, and an unmarked 100% there
    # reads as a process waiting on a peer.
    if total.on_device and total.wait_pct >= 0:
        blocked = f"{blocked} ‡"
    colour = _wait_colour(total.wait_pct)
    where = (
        f'<div class="src">{escape(_relative_to(total.origin.file, root))}'
        f":{total.origin.line}</div>"
        if total.origin is not None
        else ""
    )
    return (
        f'<tr><td class="mono">{_cell_label(total.path)}{where}</td>'
        f"<td>{total.calls:,}</td><td>{total.lanes}</td>"
        f"<td>{escape(format_ns(total.wall_ns))}</td>"
        f"<td>{escape(format_ns(total.self_ns))}</td>"
        f"<td>{blocked}</td>"
        f'<td class="barcell"><span class="bar" style="width:{width:.1f}%;'
        f'background:{colour}"></span><span class="barnum">{share:.1f}%</span></td></tr>'
    )


def _wait_colour(wait_pct: float) -> str:
    """The chart's blend as a CSS colour: reddish is working, blue is blocked.

    Duplicated from the script deliberately — the table is server-rendered and the canvas is
    not, and a reader comparing the two must never see them disagree. The constants are the
    same ones :func:`_SCRIPT`'s ``waitColour`` uses.
    """
    if wait_pct < 0:
        return "var(--muted)"
    fraction = max(0.0, min(1.0, wait_pct / 100.0))
    red = round(0xB2 + (0x2F - 0xB2) * fraction)
    green = round(0x3C + (0x6F - 0x3C) * fraction)
    blue = round(0x17 + (0x9F - 0x17) * fraction)
    return f"rgb({red},{green},{blue})"


def _gpu_device_count(payload: dict[str, JsonValue]) -> int:
    """How many device rows the chart will draw, read back off the payload it will draw from.

    Taken from the payload rather than recomputed so the note cannot claim a strip the script
    does not paint — the two would otherwise be free to disagree about an empty series.
    """
    gpu = payload.get("gpu")
    return len(gpu) if isinstance(gpu, list) else 0


def _gpu_note(devices: int) -> str:
    """Say what the device strip is, and what it is not, beside the strip itself.

    The strip is drawn under lanes of precise spans, so without a word next to it a reader
    reasonably assumes it has the same resolution and the same attribution. It has neither:
    it is a 1 Hz whole-device reading, and nothing here ties a kernel to the call that
    launched it. A reader who believes otherwise will read a busy device next to a busy lane
    as the one causing the other, which is exactly the wrong conclusion this page exists to
    prevent — and the same error, from the other direction, as the async-submission trap.
    """
    if not devices:
        return ""
    strip = "strip" if devices == 1 else "strips"
    return (
        f'<p class="note"><strong>GPU {strip}</strong> below the lanes: whole-device '
        "utilisation from the 1 Hz sampler, 0–100% per device, on the same axis. It is every "
        "process's work on that device, not only this run's, and it is far coarser than the "
        "spans above it — a kernel shorter than a second may not appear at all. Nothing "
        "connects a reading to the call that launched it: use it to ask whether the device "
        "was busy while a lane sat idle, never to attribute device time to a phase.</p>\n"
    )


def _canvas_block(gpu_devices: int = 0) -> str:
    """The timeline itself, drawn by the inlined script into a canvas."""
    # The legend is markup rather than something the script paints, so it is readable before
    # the canvas has drawn and survives the reader printing the page. What each control does
    # is on the control, not in a paragraph underneath it that gets skipped.
    legend = (
        '<div class="tl-legend">'
        '<span class="key"><i class="sw" style="background:rgb(178,60,23)"></i>'
        "on CPU</span>"
        '<span class="key"><i class="sw" style="background:rgb(47,111,159)"></i>'
        "blocked</span>"
        '<span class="key"><i class="sw grad"></i>the blend between them</span>'
        '<span class="key"><i class="sw hatch"></i>CPU not measured</span>'
        '<span class="key"><i class="sw crit"></i>critical path</span>'
        '<span class="key"><i class="sw arrow"></i>signal → the wait it released</span>'
        "</div>"
    )
    return (
        "<h2>Timeline</h2>\n"
        '<p class="note">One row per worker thread, all on the same clock. Each bar is one '
        "call; its width is wall time and its colour is how much of that time was spent "
        "blocked rather than running. A gap is a lane with nothing open at all.</p>\n"
        f"{legend}\n"
        f"{_gpu_note(gpu_devices)}"
        '<div class="tl-controls">'
        '<button type="button" id="tl-reset">reset zoom</button>'
        '<button type="button" id="tl-critical" aria-pressed="false">'
        "only the critical path</button>"
        '<button type="button" id="tl-brush" aria-pressed="false">'
        "measure a range</button>"
        '<span class="note" id="tl-range"></span>'
        "</div>\n"
        '<div id="tl-wrap"><canvas id="tl-canvas"></canvas>'
        '<div id="tl-tip" hidden></div></div>\n'
        '<div class="note" id="tl-selection"></div>\n'
        '<div class="note" id="tl-focus"></div>\n'
        '<p class="note"><strong>Drag</strong> to pan · <strong>scroll</strong> to zoom · '
        "<strong>hover</strong> for the exact figures · <strong>click a span</strong> to pin "
        "it: everything not causally upstream of it dims, and the panel names who released "
        "it · <strong>click it again</strong> to unpin. <strong>Click a lane's label</strong> "
        "to fold it to a single row — its spans stay drawn, they just stop claiming a row "
        "each. “Measure a range” answers what every lane was doing over a window you drag — "
        "the difference between a stall and a queue.</p>\n"
        '<p class="note">With the chart focused: <span class="mono">← →</span> pan · '
        '<span class="mono">+ −</span> zoom · <span class="mono">n p</span> step along the '
        'critical path · <span class="mono">0</span> reset · <span class="mono">Esc</span> '
        "unpin.</p>"
    )


def _critical_path_block(chain: list[PlacedSpan], aligned: AlignedTrace) -> str:
    """The chain that actually set the run's duration, newest first.

    The single most useful thing on the page: it converts "everything looks a bit slow" into
    an ordered list of what was waiting on what.
    """
    if not chain:
        return ""
    root = _common_root([span.origin.file for span in chain if span.origin is not None])
    rows = []
    for index, span in enumerate(chain):
        following = chain[index - 1] if index else None
        gap = ""
        if following is not None:
            idle = following.t0_ns - span.t1_ns
            if idle > 0:
                gap = f"{format_ns(idle)} idle after"
        where = (
            f'<div class="src">{escape(_relative_to(span.origin.file, root))}'
            f":{span.origin.line}</div>"
            if span.origin is not None
            else ""
        )
        rows.append(
            f"<tr><td>{escape(span.worker)}</td>"
            f"<td>{_cell_label('/'.join(span.path))}{where}</td>"
            f"<td>{escape(format_ns(span.duration_ns))}</td>"
            f"<td>{'n/a' if span.wait_pct < 0 else f'{span.wait_pct:.0f}%'}</td>"
            f"<td>{escape(gap)}</td></tr>",
        )
    note = (
        "Read bottom to top: each row had to finish before the one above it could start."
    )
    if not aligned.arrows:
        note += (
            " No signal/wait_on pairs were recorded, so this chain follows each lane's own "
            "order only — it cannot yet cross between workers."
        )
    return (
        "<h2>Critical path</h2>\n"
        '<div class="scroll"><table><thead><tr><th>worker</th><th>phase</th>'
        "<th>duration</th><th>wait</th><th>gap</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>\n"
        f'<p class="note">{escape(note)}</p>'
    )


def _caveats(aligned: AlignedTrace, omitted: int = 0) -> str:
    """Everything that makes this timeline less than the whole truth.

    A dropped span, an unmatched wait and a cross-host clock are all reasons a reader should
    trust the picture slightly less, and each is invisible unless stated.
    """
    items: list[str] = []
    if omitted:
        # The count was always computed and embedded; it just never reached the page, so the
        # one cap a reader is most likely to hit was also the only one that stayed silent.
        items.append(
            f"{omitted:,} span(s) are not drawn: the page keeps the longest spans up to its "
            "cap, so short spans in dense regions are missing. Raise --max-spans to keep "
            "more.",
        )
    if aligned.dropped_spans:
        items.append(
            f"{aligned.dropped_spans:,} span(s) were dropped: the ring buffer wrapped, so "
            "this shows the most recent activity, not the whole run. Raise "
            "trace_capacity to keep more.",
        )
    if aligned.dropped_links:
        items.append(f"{aligned.dropped_links:,} link(s) were dropped for the same reason.")
    if aligned.unmatched_waits:
        shown = ", ".join(
            f"{channel}:{key} on {worker}"
            for worker, channel, key in aligned.unmatched_waits[:5]
        )
        extra = len(aligned.unmatched_waits) - 5
        more = f" (+{extra} more)" if extra > 0 else ""
        items.append(
            f"{len(aligned.unmatched_waits)} wait_on call(s) had no matching signal, so "
            f"their arrows are missing: {shown}{more}.",
        )
    folded = sum(1 for span in aligned.spans if span.depth > _MAX_DEPTH_DRAWN)
    if folded:
        items.append(
            f"{folded:,} span(s) nest deeper than {_MAX_DEPTH_DRAWN} levels and are drawn on "
            f"the last row rather than their own, so at that depth a bar may be overlapped by "
            "a deeper one. Their times are unaffected.",
        )
    if any(span.origin is not None for span in aligned.spans):
        items.append(
            "Source locations name where a function is defined — its def line — not the line "
            "inside it that blocked. A span covers a whole call, so the line that spent the "
            "time is not recorded and is not claimed here.",
        )
    unmeasured = sum(1 for span in aligned.spans if not span.cpu_measured)
    if unmeasured:
        items.append(
            f"{unmeasured:,} span(s) were derived from function calls, which cannot measure "
            "CPU time; they are drawn hatched and their wait is unknown, not zero.",
        )
    stepped = clock_step_note(aligned.clock_steps)
    if stepped:
        items.append(stepped)
    items.append(alignment_accuracy_note(aligned.hosts, clock_stepped=bool(stepped)))
    entries = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<h2>Caveats</h2>\n<ul>{entries}</ul>"


def _payload(
    aligned: AlignedTrace,
    chain: list[PlacedSpan],
    run: MergedRun,
    max_spans: int | None = None,
) -> dict[str, JsonValue]:
    """The data the page draws from, also embedded for anything that wants to read it.

    Times are re-based to the trace's own start and expressed in microseconds: the absolute
    epoch nanosecond is both unreadable and beyond a JavaScript integer, and the offset is
    what every figure on the page is actually about.
    """
    origin = aligned.t0_ns
    drawn, omitted = _spans_to_draw(aligned.spans, max_spans)
    lane_index = {lane: index for index, lane in enumerate(aligned.lanes)}
    critical = {id(span) for span in chain}
    origins, origin_index = _origin_table(drawn)

    return {
        "run_id": str(run.metadata.get("run_id", "unknown")),
        "duration_us": aligned.duration_ns / 1000.0,
        "omitted": omitted,
        "lanes": [
            {
                "id": lane,
                "role": aligned.roles.get(lane, ""),
                "busy": round(lane_busy_share(aligned, lane), 1),
                "working": round(lane_working_share(aligned, lane), 1),
                "rows": min(max_depth_of(aligned, lane), _MAX_DEPTH_DRAWN) + 1,
            }
            for lane in aligned.lanes
        ],
        "spans": [
            {
                "l": lane_index.get(span.lane, 0),
                "n": "/".join(span.path),
                "t": round((span.t0_ns - origin) / 1000.0, 3),
                "d": round(span.duration_ns / 1000.0, 3),
                "w": round(span.wait_pct, 1),
                "y": min(span.depth, _MAX_DEPTH_DRAWN),
                "a": 1 if span.flags & FLAG_AUTO else 0,
                "c": 1 if id(span) in critical else 0,
                "o": origin_index.get(span.origin, -1),
            }
            for span in drawn
        ],
        "origins": origins,
        "arrows": [
            {
                "s": arrow.src_worker,
                "d": arrow.dst_worker,
                "t0": round((arrow.src_t_ns - origin) / 1000.0, 3),
                "t1": round((arrow.dst_t_ns - origin) / 1000.0, 3),
                "ch": f"{arrow.channel}:{arrow.key}",
            }
            for arrow in aligned.arrows
        ],
        "gpu": _gpu_series(run, aligned),
    }


def _origin_table(
    spans: list[PlacedSpan],
) -> tuple[list[JsonValue], dict[Origin | None, int]]:
    """Intern the distinct source locations the drawn spans refer to.

    A table plus an index per span, not a location per span: the same function appears in
    thousands of spans, and repeating its path in each would dominate the page. Spans with no
    origin — every named phase — map to ``-1`` and render without a location rather than with
    an invented one.

    Paths are shortened against their common root so the tooltip shows ``envs/env.py`` rather
    than a hundred characters of absolute path that pushes the useful part off the edge. The
    root is reported alongside, so a shortened path is still resolvable to a real file.
    """
    distinct = [span.origin for span in spans if span.origin is not None]
    unique: list[Origin] = list(dict.fromkeys(distinct))
    if not unique:
        return [], {}

    root = _common_root([item.file for item in unique])
    index: dict[Origin | None, int] = {item: position for position, item in enumerate(unique)}
    table: list[JsonValue] = [
        {
            "f": _relative_to(item.file, root),
            "n": item.function,
            "l": item.line,
            "full": item.file,
        }
        for item in unique
    ]
    return table, index


def _common_root(files: list[str]) -> str:
    """The deepest directory every file shares, or ``""`` when they share none.

    Always the common *directory*, never a file: several functions of one module give
    ``commonpath`` a list whose entries are all the same path, and it returns that file — so
    every location would shorten to ``.`` and the module name would vanish from the page.

    Guarded rather than trusted: ``commonpath`` raises on a mix of absolute and relative
    paths, which a run spanning a real module and a ``<string>`` entry point produces.
    Falling back to no shortening shows longer paths, never wrong ones.
    """
    if not files:
        return ""
    try:
        common = os.path.commonpath(files)
    except ValueError:
        return ""
    if common in files:
        return os.path.dirname(common)
    return common


def _relative_to(file: str, root: str) -> str:
    """``file`` shown against ``root``, falling back to the full path when it is not under it."""
    if not root:
        return file
    try:
        return os.path.relpath(file, root)
    except ValueError:
        return file


def _spans_to_draw(
    spans: list[PlacedSpan],
    max_spans: int | None = None,
) -> tuple[list[PlacedSpan], int]:
    """Cap what the page carries, keeping the longest spans and counting the rest.

    The longest rather than the first: a reader zooming into a busy region wants the shape of
    it, and the spans that define that shape are the ones with visible width. The number
    dropped is reported, never hidden.
    """
    limit = _MAX_SPANS_DRAWN if max_spans is None else max(1, max_spans)
    if len(spans) <= limit:
        return spans, 0
    kept = sorted(spans, key=lambda span: -span.duration_ns)[:limit]
    kept.sort(key=lambda span: span.t0_ns)
    return kept, len(spans) - limit


def _gpu_series(run: MergedRun, aligned: AlignedTrace) -> JsonValue:
    """Per-device utilisation over the same axis, so an idle lane can be checked against it.

    Drawn from the 1 Hz sampler, so it is coarse next to the spans — but the question it
    answers is coarse too: was the GPU busy while the CPU lanes were empty?
    """
    analysis = analyse_processes(run.samples_by_process())
    if not analysis.gpu_devices:
        return []
    origin_s = aligned.t0_ns / 1e9
    series: list[JsonValue] = []
    for device in analysis.gpu_devices:
        points: list[JsonValue] = [
            [round(at - origin_s, 2), round(value, 1)]
            for at, value in _device_samples(run, device.index)
        ]
        if points:
            series.append({"device": device.index, "points": points})
    return series


def _device_samples(run: MergedRun, index: int) -> list[tuple[float, float]]:
    """Utilisation readings for one device across every worker, in time order.

    ``samples_by_process`` keeps each worker's rows separate because cumulative counters may
    only be differenced within one process. Utilisation is not cumulative — it is an
    instantaneous reading — so pooling it across workers is safe here, and is what puts every
    device on one line.
    """
    readings: list[tuple[float, float]] = []
    for samples in run.samples_by_process():
        for sample in samples:
            value = sample.gpu_utils.get(index)
            if value is not None and value >= 0:
                readings.append((sample.t, float(value)))
    readings.sort()
    return readings


_STYLE = """
.warn {
  background: var(--panel); border: 1px solid var(--rule);
  border-left: 3px solid var(--accent); border-radius: 6px;
  padding: .55rem .8rem; margin: .7rem 0 0; font-size: .87rem;
}
.findings { list-style: none; counter-reset: finding; margin: 0; padding: 0; }
.finding {
  counter-increment: finding; position: relative; background: var(--panel);
  border: 1px solid var(--rule); border-left: 3px solid var(--accent); border-radius: 6px;
  padding: .7rem .9rem .7rem 2.6rem; margin: 0 0 .5rem;
}
.finding::before {
  content: counter(finding); position: absolute; left: .9rem; top: .7rem;
  font-variant-numeric: tabular-nums; font-weight: 600; color: var(--accent);
}
.finding .claim { font-weight: 600; }
.finding .why { color: var(--muted); font-size: .87rem; margin-top: .2rem; }
.finding .act:not(:empty) { margin-top: .45rem; }
.finding[data-kind="idle-lane"] { border-left-color: var(--cool); }
.finding[data-kind="idle-lane"]::before { color: var(--cool); }
.tl-jump {
  font: inherit; font-size: .78rem; padding: .2rem .55rem; cursor: pointer;
  background: var(--bg); color: var(--fg); border: 1px solid var(--rule); border-radius: 5px;
}
.tl-jump:hover { border-color: var(--accent); color: var(--accent); }
.barcell { min-width: 9rem; }
.bar {
  display: inline-block; height: .62rem; border-radius: 2px; vertical-align: middle;
  min-width: 2px;
}
.barnum { color: var(--muted); font-size: .78rem; margin-left: .4rem; }
.tl-legend {
  display: flex; flex-wrap: wrap; gap: .5rem 1.1rem; margin: 0 0 .7rem;
  font-size: .8rem; color: var(--muted);
}
.tl-legend .key { display: inline-flex; align-items: center; gap: .35rem; }
.tl-legend .sw {
  display: inline-block; width: .95rem; height: .62rem; border-radius: 2px;
}
.tl-legend .grad {
  background: linear-gradient(90deg, rgb(178,60,23), rgb(47,111,159)); width: 2.4rem;
}
.tl-legend .hatch {
  background: repeating-linear-gradient(
    45deg, var(--muted) 0 1px, transparent 1px 4px
  );
  border: 1px solid var(--rule);
}
.tl-legend .crit { background: transparent; border: 2px solid var(--accent); }
.tl-legend .arrow { background: var(--cool); height: 2px; border-radius: 0; }
#tl-focus:not(:empty) {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
  padding: .5rem .7rem; margin-top: .5rem;
}
#tl-focus .lead { color: var(--fg); font-weight: 600; }
.tl-controls { display: flex; gap: .6rem; align-items: center; margin: 0 0 .6rem; }
.tl-controls button {
  font: inherit; font-size: .8rem; padding: .25rem .6rem; cursor: pointer;
  background: var(--panel); color: var(--fg); border: 1px solid var(--rule);
  border-radius: 5px;
}
.tl-controls button:hover { border-color: var(--muted); }
.tl-controls button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
#tl-wrap { position: relative; border: 1px solid var(--rule); border-radius: 6px; }
#tl-canvas { display: block; width: 100%; cursor: grab; }
#tl-canvas.dragging { cursor: grabbing; }
#tl-tip {
  position: absolute; pointer-events: none; z-index: 5; max-width: 26rem;
  background: var(--bg); color: var(--fg); border: 1px solid var(--rule);
  border-radius: 5px; padding: .4rem .55rem; font-size: .8rem; line-height: 1.4;
  box-shadow: 0 2px 10px rgba(0,0,0,.18); font-variant-numeric: tabular-nums;
}
#tl-tip .p { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
#tl-tip .crit { color: var(--accent); font-weight: 600; }
#tl-tip .src, #tl-focus .src {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .75rem; color: var(--muted); word-break: break-all;
}
/* The same location line inside a table cell: secondary to the phase name above it, so it
   reads as an annotation on that row rather than as another column. */
td .src {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .72rem; color: var(--muted); font-weight: 400; margin-top: .1rem;
}
"""

# Vanilla, inlined, no build step and no dependency. Kept deliberately small: pan, zoom,
# hover, and highlight a causal chain. Anything more belongs in Perfetto.
_SCRIPT = r"""
(function () {
  var el = document.getElementById('tl-canvas');
  if (!el) { return; }
  var raw = document.getElementById('lineprofiler-data');
  if (!raw) { return; }
  var data = JSON.parse(raw.textContent.replace(/<\\\//g, '</'));
  var spans = data.spans || [], lanes = data.lanes || [], arrows = data.arrows || [];
  var origins = data.origins || [];
  var gpu = data.gpu || [];
  if (!spans.length) { return; }

  var ctx = el.getContext('2d');
  var tip = document.getElementById('tl-tip');
  var wrap = document.getElementById('tl-wrap');
  var rangeLabel = document.getElementById('tl-range');
  var ROW_H = 15, LANE_PAD = 8, PAD_L = 132, PAD_T = 26, GPU_H = 34;
  var total = data.duration_us || 1;
  var view = { t0: 0, t1: total };
  var onlyCritical = false, hover = null, focus = null, chain = null;
  var brushing = false, brushFrom = null, brushTo = null;
  var selectionLabel = document.getElementById('tl-selection');
  var focusPanel = document.getElementById('tl-focus');

  // Each lane is as tall as it is deep: one row per nesting level, so a callee is drawn
  // under its caller instead of over it. Every y on the page reads these tables.
  //
  // Recomputed rather than computed once, because a lane can now be collapsed: with sixteen
  // actors the chart is taller than any screen and the reader cannot see the learner and an
  // actor at the same time, which is exactly the comparison they opened it to make. A
  // collapsed lane keeps its slot and its spans — they fold onto one row — so collapsing
  // never hides activity, it only stops giving it a row of its own.
  var laneTop = [], laneH = [], lanesBottom = PAD_T;
  var collapsed = {};
  function layout() {
    lanesBottom = PAD_T;
    for (var li = 0; li < lanes.length; li++) {
      var rows = collapsed[li] ? 1 : (lanes[li].rows || 1);
      laneTop[li] = lanesBottom;
      laneH[li] = rows * ROW_H + LANE_PAD;
      lanesBottom += laneH[li];
    }
  }
  layout();

  // Which row a span lands on: its own depth normally, row 0 when its lane is folded.
  function rowOf(sp) {
    return collapsed[sp.l] ? 0 : (sp.y || 0);
  }

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function height() {
    return lanesBottom + gpu.length * GPU_H + 18;
  }
  function resize() {
    var ratio = window.devicePixelRatio || 1;
    var width = wrap.clientWidth;
    el.width = width * ratio; el.height = height() * ratio;
    el.style.height = height() + 'px';
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }
  function xOf(t, width) {
    return PAD_L + (t - view.t0) / (view.t1 - view.t0) * (width - PAD_L - 10);
  }
  function tOf(x, width) {
    return view.t0 + (x - PAD_L) / (width - PAD_L - 10) * (view.t1 - view.t0);
  }

  // The same blend the icicle chart uses, so the two pages agree on what colour means.
  function waitColour(w) {
    if (w < 0) { return null; }
    var f = Math.max(0, Math.min(1, w / 100));
    var r = Math.round(0xB2 + (0x2F - 0xB2) * f);
    var g = Math.round(0x3C + (0x6F - 0x3C) * f);
    var b = Math.round(0x17 + (0x9F - 0x17) * f);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  // The only way text reaches the page: textContent never parses markup, so a phase name is
  // always shown rather than interpreted.
  function named(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    node.textContent = text;
    return node;
  }

  // Spans index into the origin table rather than carrying a path each: the same function
  // appears in thousands of them. -1, and any index the table does not hold, means no
  // location was recorded — a named phase has no code object behind it.
  function originOf(sp) {
    var at = sp.o;
    if (at === undefined || at < 0 || at >= origins.length) { return null; }
    return origins[at];
  }

  function fmt(us) {
    if (us < 1) { return (us * 1000).toFixed(0) + ' ns'; }
    if (us < 1000) { return us.toFixed(1) + ' µs'; }
    if (us < 1e6) { return (us / 1000).toFixed(2) + ' ms'; }
    return (us / 1e6).toFixed(2) + ' s';
  }

  function draw() {
    var width = wrap.clientWidth;
    ctx.clearRect(0, 0, width, height());
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    var rule = css('--rule'), muted = css('--muted'), fg = css('--fg');

    drawAxis(width, muted, rule);
    for (var i = 0; i < lanes.length; i++) { drawLaneLabel(i, width, fg, muted, rule); }
    for (var s = 0; s < spans.length; s++) { drawSpan(spans[s], width); }
    drawArrows(width);
    drawGpu(width, muted, rule);
    drawBrush(width, fg);
    rangeLabel.textContent = fmt(view.t1 - view.t0) + ' shown of ' + fmt(total);
  }

  function drawBrush(width, fg) {
    if (brushFrom === null || brushTo === null) { return; }
    var x0 = xOf(Math.min(brushFrom, brushTo), width);
    var x1 = xOf(Math.max(brushFrom, brushTo), width);
    if (x1 - x0 < 1) { return; }
    // Drawn last so it sits over the spans it selects, and translucent so they stay readable:
    // the point is to say which spans are included, not to hide them.
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = fg;
    ctx.fillRect(x0, PAD_T, x1 - x0, lanesBottom - PAD_T);
    ctx.restore();
    ctx.strokeStyle = fg; ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, PAD_T + 0.5, x1 - x0 - 1, lanesBottom - PAD_T - 1);
  }

  function describeSelection() {
    var from = Math.min(brushFrom, brushTo), to = Math.max(brushFrom, brushTo);
    var window = to - from;
    if (window <= 0) { selectionLabel.textContent = ''; return; }

    // Union per lane, so a lane whose phases nest cannot exceed 100% of the window.
    var byLane = {};
    for (var i = 0; i < spans.length; i++) {
      var sp = spans[i];
      var lo = Math.max(sp.t, from), hi = Math.min(sp.t + sp.d, to);
      if (hi <= lo) { continue; }
      var name = lanes[sp.l] ? lanes[sp.l].id : String(sp.l);
      if (!byLane[name]) { byLane[name] = []; }
      byLane[name].push([lo, hi]);
    }

    var parts = [];
    for (var lane in byLane) {
      if (!Object.prototype.hasOwnProperty.call(byLane, lane)) { continue; }
      parts.push([lane, 100 * covered(byLane[lane]) / window]);
    }
    parts.sort(function (a, b) { return b[1] - a[1]; });

    var text = 'during ' + fmt(window) + ': ';
    if (!parts.length) {
      text += 'no lane had a phase open — a stall, not a queue.';
    } else {
      var shown = [];
      for (var k = 0; k < parts.length && k < 6; k++) {
        shown.push(parts[k][0] + ' busy ' + parts[k][1].toFixed(0) + '%');
      }
      text += shown.join(', ');
    }
    selectionLabel.textContent = text;
  }

  function covered(intervals) {
    intervals.sort(function (a, b) { return a[0] - b[0]; });
    var total = 0, cursor = -Infinity;
    for (var i = 0; i < intervals.length; i++) {
      var start = Math.max(intervals[i][0], cursor);
      if (intervals[i][1] > start) { total += intervals[i][1] - start; cursor = intervals[i][1]; }
    }
    return total;
  }

  function drawAxis(width, muted, rule) {
    ctx.strokeStyle = rule; ctx.fillStyle = muted; ctx.lineWidth = 1;
    var span = view.t1 - view.t0;
    var step = Math.pow(10, Math.floor(Math.log10(span / 6)));
    if (span / step > 12) { step *= 5; } else if (span / step > 6) { step *= 2; }
    ctx.textAlign = 'center';
    for (var t = Math.ceil(view.t0 / step) * step; t < view.t1; t += step) {
      var x = xOf(t, width);
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(x, PAD_T - 6); ctx.lineTo(x, height() - 14); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText(fmt(t), x, 10);
    }
    ctx.textAlign = 'left';
  }

  function drawLaneLabel(i, width, fg, muted, rule) {
    var y = laneTop[i];
    ctx.fillStyle = fg;
    // The disclosure marker doubles as the affordance: a lane that can be folded looks
    // foldable, and one already folded says so without the reader having to count rows.
    var marker = (lanes[i].rows || 1) > 1 ? (collapsed[i] ? '▸ ' : '▾ ') : '  ';
    var label = marker + lanes[i].id;
    if (label.length > 20) { label = label.slice(0, 19) + '…'; }
    ctx.fillText(label, 6, y + 9);
    ctx.fillStyle = muted;
    ctx.fillText(lanes[i].role, 6, y + 21);
    ctx.strokeStyle = rule; ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(PAD_L, y + laneH[i]); ctx.lineTo(width - 10, y + laneH[i]);
    ctx.stroke(); ctx.globalAlpha = 1;
  }

  function drawSpan(sp, width) {
    if (onlyCritical && !sp.c) { return; }
    // Dimmed, never hidden: a reader tracing a chain still needs to see what else was running
    // to judge whether the gap around it was a queue or an empty machine.
    var offChain = chain && !chain[spanKey(sp)];
    if (offChain) { ctx.globalAlpha = 0.15; }
    var x0 = xOf(sp.t, width), x1 = xOf(sp.t + sp.d, width);
    if (x1 < PAD_L || x0 > width) { return; }
    x0 = Math.max(x0, PAD_L);
    var w = Math.max(x1 - x0, 1);
    var y = laneTop[sp.l] + rowOf(sp) * ROW_H + 3, h = ROW_H - 3;
    var colour = waitColour(sp.w);
    if (colour === null) {
      // Unmeasured CPU: hatched, so it can never be mistaken for a span that never waited.
      ctx.fillStyle = css('--muted'); ctx.globalAlpha = 0.28;
      ctx.fillRect(x0, y, w, h); ctx.globalAlpha = 1;
      ctx.strokeStyle = css('--muted'); ctx.globalAlpha = 0.5;
      ctx.beginPath();
      for (var hx = x0; hx < x0 + w; hx += 5) {
        ctx.moveTo(hx, y + h); ctx.lineTo(Math.min(hx + h, x0 + w), y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = colour;
      ctx.fillRect(x0, y, w, h);
    }
    if (sp.c) {
      ctx.strokeStyle = css('--accent'); ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, y + 1, w - 2, h - 2); ctx.lineWidth = 1;
    }
    if (focus && focus === sp) {
      ctx.strokeStyle = css('--fg'); ctx.lineWidth = 2;
      ctx.strokeRect(x0, y, w, h); ctx.lineWidth = 1;
    }
    if (w > 34 && !offChain) {
      var name = sp.n.split('/').pop();
      if (name.length * 6 < w - 6) {
        ctx.fillStyle = '#ffffff';
        ctx.fillText(name, x0 + 4, y + h / 2);
      }
    }
    ctx.globalAlpha = 1;
  }

  function laneOf(worker) {
    for (var i = 0; i < lanes.length; i++) {
      if (lanes[i].id.indexOf(worker + '#') === 0) { return i; }
    }
    return -1;
  }

  function drawArrows(width) {
    if (!arrows.length) { return; }
    ctx.strokeStyle = css('--cool'); ctx.fillStyle = css('--cool');
    for (var i = 0; i < arrows.length; i++) {
      var a = arrows[i];
      var from = laneOf(a.s), to = laneOf(a.d);
      if (from < 0 || to < 0) { continue; }
      var x0 = xOf(a.t0, width), x1 = xOf(a.t1, width);
      if (x1 < PAD_L || x0 > width) { continue; }
      // Anchored to each lane's top row: that is the phase-open row, and it keeps arrows
      // readable as a lane grows rows rather than burying them among nested bars.
      var y0 = laneTop[from] + ROW_H / 2;
      var y1 = laneTop[to] + ROW_H / 2;
      ctx.globalAlpha = focus ? 0.25 : 0.55;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.bezierCurveTo(x0 + 14, y0, x1 - 14, y1, x1, y1);
      ctx.stroke();
      var dir = y1 > y0 ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - 4, y1 - 5 * dir); ctx.lineTo(x1 + 4, y1 - 5 * dir);
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  function drawGpu(width, muted, rule) {
    if (!gpu.length) { return; }
    var base = lanesBottom;
    for (var d = 0; d < gpu.length; d++) {
      var top = base + d * GPU_H;
      ctx.fillStyle = muted;
      ctx.fillText('GPU ' + gpu[d].device, 6, top + GPU_H / 2);
      var pts = gpu[d].points;
      ctx.strokeStyle = css('--cool'); ctx.beginPath();
      var started = false;
      for (var i = 0; i < pts.length; i++) {
        var x = xOf(pts[i][0] * 1e6, width);
        var y = top + GPU_H - 6 - (pts[i][1] / 100) * (GPU_H - 12);
        if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
      }
      ctx.stroke();
      ctx.strokeStyle = rule; ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(PAD_L, top + GPU_H); ctx.lineTo(width - 10, top + GPU_H);
      ctx.stroke(); ctx.globalAlpha = 1;
    }
  }

  // Which lane owns this pixel, given lanes are no longer a uniform height.
  function laneAtY(py) {
    for (var i = 0; i < lanes.length; i++) {
      if (py >= laneTop[i] && py < laneTop[i] + laneH[i]) { return i; }
    }
    return -1;
  }

  function spanAt(px, py) {
    var width = wrap.clientWidth;
    var lane = laneAtY(py);
    if (lane < 0) { return null; }
    // Now that a nested span has its own row, the row under the cursor identifies it
    // exactly. The old innermost-wins guess existed only because parent and child shared
    // a row, and it made a parent bar unhoverable.
    var row = Math.floor((py - laneTop[lane]) / ROW_H);
    var t = tOf(px, width);
    for (var i = 0; i < spans.length; i++) {
      var sp = spans[i];
      if (sp.l !== lane || rowOf(sp) !== row) { continue; }
      if (sp.t <= t && sp.t + sp.d >= t) { return sp; }
    }
    return null;
  }

  el.addEventListener('mousemove', function (event) {
    if (dragging) { return; }
    var box = el.getBoundingClientRect();
    var sp = spanAt(event.clientX - box.left, event.clientY - box.top);
    hover = sp;
    if (!sp) { tip.hidden = true; return; }
    var waitText = sp.w < 0 ? 'unknown (derived from a function call)' : sp.w.toFixed(0) + '%';
    // Built as DOM nodes, never as an HTML string. Phase names come from user code, so a
    // name containing an image tag with an error handler would otherwise run that handler in
    // whoever opens the report — and a profiling artifact gets mailed around and opened by
    // other people.
    tip.textContent = '';
    tip.appendChild(named('div', 'p', sp.n));
    tip.appendChild(named('div', '', lanes[sp.l].id + ' · ' + lanes[sp.l].role));
    tip.appendChild(named('div', '', 'duration ' + fmt(sp.d)));
    tip.appendChild(named('div', '', 'waiting ' + waitText));
    var org = originOf(sp);
    if (org) {
      // The function's definition site, not the line that blocked: a span covers a whole
      // call, so naming one line inside it would claim more than was measured.
      tip.appendChild(named('div', 'src', org.n + '()'));
      tip.appendChild(named('div', 'src', org.f + ':' + org.l));
    }
    if (sp.c) { tip.appendChild(named('div', 'crit', 'on the critical path')); }
    tip.hidden = false;
    var x = event.clientX - box.left + 14, y = event.clientY - box.top + 14;
    if (x + tip.offsetWidth > box.width) { x = box.width - tip.offsetWidth - 6; }
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });
  el.addEventListener('mouseleave', function () { tip.hidden = true; hover = null; });
  el.addEventListener('click', function (event) {
    // The label gutter and the chart answer different clicks. Folding a lane from its own
    // label is where a reader looks for it; putting it in the toolbar would mean naming the
    // lane in a menu, which is the thing they were already pointing at.
    var box = el.getBoundingClientRect();
    if (event.clientX - box.left < PAD_L) {
      var lane = laneAtY(event.clientY - box.top);
      if (lane >= 0 && (lanes[lane].rows || 1) > 1) {
        collapsed[lane] = !collapsed[lane];
        layout();
        resize();
      }
      return;
    }
    focus = (focus === hover) ? null : hover;
    describeFocus();
    draw();
  });

  // What a click is *for*. The outline alone told the reader their click had registered and
  // nothing else; the question they clicked to ask was "why was this span waiting", and the
  // trace already knows — the arrows name the producer, and the other lanes say what was
  // running. Answering it here is what makes the timeline traceable rather than decorative.
  function describeFocus() {
    focusPanel.textContent = '';
    chain = focus ? causalChain(focus) : null;
    if (!focus) { return; }
    var lane = lanes[focus.l] || { id: '?', role: '' };
    focusPanel.appendChild(named('div', 'lead', focus.n + ' on ' + lane.id));

    var line = fmt(focus.d) + ' wall';
    if (focus.w >= 0) {
      line += ', ' + focus.w.toFixed(0) + '% of it blocked';
    } else {
      line += ', CPU time not measured so its wait is unknown';
    }
    focusPanel.appendChild(named('div', '', line));

    // The full path here rather than the shortened one: this panel is what a reader copies
    // into an editor, and a path relative to a root they have to infer is not that.
    var org = originOf(focus);
    if (org) {
      focusPanel.appendChild(named('div', 'src',
        'defined at ' + (org.full || org.f) + ':' + org.l + ' in ' + org.n + '()'));
    }

    var release = releasingArrow(focus);
    if (release) {
      focusPanel.appendChild(named('div', '',
        'released by ' + release.s + ' via ' + release.ch + ', ' +
        fmt(Math.max(0, release.t1 - release.t0)) + ' before this span ended'));
    } else if (focus.w > 50) {
      focusPanel.appendChild(named('div', '',
        'no signal/wait_on pair covers this wait, so who released it was not recorded'));
    }

    var busy = concurrentLanes(focus);
    if (busy.length) {
      focusPanel.appendChild(named('div', '', 'while it ran: ' + busy.join(', ')));
    } else {
      focusPanel.appendChild(named('div', '',
        'no other lane had a phase open while it ran — a stall, not a queue'));
    }
    focusPanel.appendChild(named('div', '', 'click the span again to unpin.'));
  }

  // Every span causally upstream of the focused one, walked back through the arrows: the
  // producer that released it, whatever released *that*, and so on. This is what "trace what
  // it was waiting for" was always supposed to mean — a pinned span with a dimmed background
  // shows the chain that led to it rather than just the span itself.
  //
  // Bounded by a visited set, because a cycle in recorded links is not hypothetical: two
  // workers that each wait on the other produce one, and an unbounded walk would hang the
  // page rather than draw a wrong picture.
  function causalChain(sp) {
    if (!sp) { return null; }
    var members = {};
    var queue = [sp];
    var guard = 0;
    members[spanKey(sp)] = true;
    while (queue.length && guard++ < 4000) {
      var current = queue.shift();
      var release = releasingArrow(current);
      if (!release) { continue; }
      // The producer's span is whatever was open on its lane when it signalled.
      for (var i = 0; i < spans.length; i++) {
        var other = spans[i];
        var lane = lanes[other.l];
        if (!lane || lane.id.split('#')[0] !== release.s) { continue; }
        if (other.t > release.t0 || other.t + other.d < release.t0) { continue; }
        var key = spanKey(other);
        if (members[key]) { continue; }
        members[key] = true;
        queue.push(other);
      }
    }
    return members;
  }

  function spanKey(sp) {
    return sp.l + ':' + sp.t + ':' + sp.n;
  }

  // The arrow that landed on this span's lane during its life, newest first: that is the
  // signal the waiter was released by.
  function releasingArrow(sp) {
    var lane = lanes[sp.l];
    if (!lane) { return null; }
    var worker = lane.id.split('#')[0];
    var best = null;
    for (var i = 0; i < arrows.length; i++) {
      var a = arrows[i];
      if (a.d !== worker) { continue; }
      if (a.t1 < sp.t || a.t1 > sp.t + sp.d) { continue; }
      if (!best || a.t1 > best.t1) { best = a; }
    }
    return best;
  }

  // Which other lanes had a phase open while this span ran, as a share of its duration. The
  // stall-versus-queue question for one span, answered without the reader dragging a range.
  function concurrentLanes(sp) {
    var from = sp.t, to = sp.t + sp.d;
    if (to <= from) { return []; }
    var byLane = {};
    for (var i = 0; i < spans.length; i++) {
      var other = spans[i];
      if (other.l === sp.l) { continue; }
      var lo = Math.max(other.t, from), hi = Math.min(other.t + other.d, to);
      if (hi <= lo) { continue; }
      var key = lanes[other.l] ? lanes[other.l].id : String(other.l);
      if (!byLane[key]) { byLane[key] = []; }
      byLane[key].push([lo, hi]);
    }
    var parts = [];
    for (var name in byLane) {
      if (!Object.prototype.hasOwnProperty.call(byLane, name)) { continue; }
      parts.push([name, 100 * covered(byLane[name]) / (to - from)]);
    }
    parts.sort(function (a, b) { return b[1] - a[1]; });
    var shown = [];
    for (var k = 0; k < parts.length && k < 4; k++) {
      shown.push(parts[k][0] + ' ' + parts[k][1].toFixed(0) + '%');
    }
    return shown;
  }

  var dragging = false, dragX = 0, dragT = 0;
  el.addEventListener('mousedown', function (event) {
    if (brushing) {
      // A brush and a pan both start with a press on the canvas, so the mode decides which.
      var rect = el.getBoundingClientRect();
      brushFrom = tOf(event.clientX - rect.left, wrap.clientWidth);
      brushTo = brushFrom;
      tip.hidden = true;
      draw();
      return;
    }
    dragging = true; dragX = event.clientX; dragT = view.t0;
    el.classList.add('dragging'); tip.hidden = true;
  });
  window.addEventListener('mouseup', function () {
    dragging = false; el.classList.remove('dragging');
    if (brushFrom !== null && brushTo !== null) { describeSelection(); }
  });
  window.addEventListener('mousemove', function (event) {
    if (brushing && brushFrom !== null) {
      var box = el.getBoundingClientRect();
      brushTo = tOf(event.clientX - box.left, wrap.clientWidth);
      draw();
      return;
    }
    if (!dragging) { return; }
    var width = wrap.clientWidth;
    var perPx = (view.t1 - view.t0) / (width - PAD_L - 10);
    var shift = (event.clientX - dragX) * perPx;
    var span = view.t1 - view.t0;
    view.t0 = dragT - shift;
    view.t1 = view.t0 + span;
    draw();
  });
  el.addEventListener('wheel', function (event) {
    event.preventDefault();
    var width = wrap.clientWidth;
    var box = el.getBoundingClientRect();
    var anchor = tOf(event.clientX - box.left, width);
    var factor = event.deltaY > 0 ? 1.25 : 0.8;
    var span = (view.t1 - view.t0) * factor;
    if (span > total * 4 || span < 0.05) { return; }
    var ratio = (anchor - view.t0) / (view.t1 - view.t0);
    view.t0 = anchor - span * ratio;
    view.t1 = view.t0 + span;
    draw();
  }, { passive: false });

  document.getElementById('tl-reset').addEventListener('click', function () {
    view.t0 = 0; view.t1 = total; focus = null;
    describeFocus();
    draw();
  });

  // The findings section names a lane or a phase; these buttons carry that name down to the
  // chart. Zooming to the widest matching span rather than merely highlighting it, because a
  // finding about a phase buried at 4% of the way through a five-minute run is not visible
  // at full zoom, and a reader told to "look at queue_get" with no way to get there is being
  // given a homework assignment rather than an answer.
  function jumpTo(anchor) {
    var best = null;
    for (var i = 0; i < spans.length; i++) {
      var sp = spans[i];
      var lane = lanes[sp.l] ? lanes[sp.l].id : '';
      if (sp.n !== anchor && lane !== anchor) { continue; }
      if (!best || sp.d > best.d) { best = sp; }
    }
    if (!best) { return; }
    var pad = Math.max(best.d * 0.5, total * 0.01);
    view.t0 = best.t - pad;
    view.t1 = best.t + best.d + pad;
    focus = best;
    describeFocus();
    draw();
    wrap.scrollIntoView({ block: 'center' });
  }

  var jumps = document.getElementsByClassName('tl-jump');
  for (var j = 0; j < jumps.length; j++) {
    jumps[j].addEventListener('click', function (event) {
      jumpTo(event.currentTarget.getAttribute('data-anchor'));
    });
  }
  var criticalButton = document.getElementById('tl-critical');
  criticalButton.addEventListener('click', function () {
    onlyCritical = !onlyCritical;
    criticalButton.setAttribute('aria-pressed', onlyCritical ? 'true' : 'false');
    draw();
  });
  var brushButton = document.getElementById('tl-brush');
  brushButton.addEventListener('click', function () {
    brushing = !brushing;
    brushButton.setAttribute('aria-pressed', brushing ? 'true' : 'false');
    if (!brushing) { brushFrom = null; brushTo = null; selectionLabel.textContent = ''; }
    draw();
  });

  // Keyboard navigation. A canvas is not reachable by keyboard at all without this, so the
  // whole chart was mouse-only — and stepping the critical path one span at a time is the
  // fastest way to read it even with a mouse, because each step re-centres and re-pins.
  //
  // Bound to the canvas rather than the document, and only after it is focused, so arrow keys
  // still scroll the page while the reader is anywhere else on it.
  el.tabIndex = 0;
  el.addEventListener('keydown', function (event) {
    var span = view.t1 - view.t0;
    var handled = true;
    if (event.key === 'ArrowLeft') {
      view.t0 -= span * 0.15; view.t1 -= span * 0.15;
    } else if (event.key === 'ArrowRight') {
      view.t0 += span * 0.15; view.t1 += span * 0.15;
    } else if (event.key === '+' || event.key === '=') {
      zoomBy(0.8);
    } else if (event.key === '-' || event.key === '_') {
      zoomBy(1.25);
    } else if (event.key === 'n' || event.key === 'p') {
      stepCritical(event.key === 'n' ? 1 : -1);
    } else if (event.key === 'Escape') {
      focus = null; chain = null; describeFocus();
    } else if (event.key === '0') {
      view.t0 = 0; view.t1 = total;
    } else {
      handled = false;
    }
    if (handled) { event.preventDefault(); draw(); }
  });

  function zoomBy(factor) {
    var middle = (view.t0 + view.t1) / 2;
    var span = (view.t1 - view.t0) * factor;
    if (span > total * 4 || span < 0.05) { return; }
    view.t0 = middle - span / 2;
    view.t1 = middle + span / 2;
  }

  // Walking the critical path in recorded order, centring each span. The chain is the ordered
  // answer to "what set this run's length", and reading it off the chart otherwise means
  // hunting for outlined bars across lanes that may be far apart vertically.
  function stepCritical(direction) {
    var path = [];
    for (var i = 0; i < spans.length; i++) {
      if (spans[i].c) { path.push(spans[i]); }
    }
    if (!path.length) { return; }
    path.sort(function (a, b) { return a.t - b.t; });
    var index = -1;
    for (var k = 0; k < path.length; k++) {
      if (path[k] === focus) { index = k; break; }
    }
    index = (index + direction + path.length) % path.length;
    var target = path[index];
    focus = target;
    var pad = Math.max(target.d * 0.6, total * 0.005);
    view.t0 = target.t - pad;
    view.t1 = target.t + target.d + pad;
    describeFocus();
  }

  window.addEventListener('resize', resize);
  resize();
})();
"""
