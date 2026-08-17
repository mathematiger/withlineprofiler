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

from pathlib import Path

from lineprofiler.accounting.analysis import analyse_processes
from lineprofiler.accounting.report import format_ns
from lineprofiler.accounting.snapshot import MergedRun
from lineprofiler.accounting.trace import FLAG_AUTO
from lineprofiler.accounting.tracealign import (
    AlignedTrace,
    PlacedSpan,
    align_run,
    alignment_accuracy_note,
    critical_path,
    lane_busy_share,
    lane_working_share,
    max_depth_of,
)
from lineprofiler.htmldoc import JsonValue, document, escape, tile

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


def render_trace_html(run: MergedRun, title: str = "lineprofiler trace") -> str:
    """Return a complete, self-contained timeline page for a merged run.

    Raises nothing when a run has no trace data: the page says so plainly and explains how to
    record some, which is more useful than an exception to someone who has just discovered
    the feature exists.
    """
    aligned = align_run(run)
    if not aligned.spans:
        return document(title, _empty_body(title), data={"spans": []})

    chain = critical_path(aligned)
    payload = _payload(aligned, chain, run)
    body = "\n".join([
        _header(aligned, title, run),
        _lane_summary(aligned),
        _canvas_block(),
        _critical_path_block(chain, aligned),
        _sequence_block(aligned),
        _caveats(aligned),
    ])
    return document(title, body, data=payload, style=_STYLE, script=_SCRIPT)


def write_trace_html(
    run: MergedRun,
    path: str | Path,
    title: str = "lineprofiler trace",
) -> None:
    """Write :func:`render_trace_html` to ``path``, creating parent directories if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_trace_html(run, title), encoding="utf-8")


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
    """Headline figures: how long, how many lanes, and how much of it was waiting."""
    waiting = _overall_wait_share(aligned)
    tiles = "".join([
        tile("span", format_ns(aligned.duration_ns)),
        tile("lanes", str(len(aligned.lanes))),
        tile("spans", f"{len(aligned.spans):,}"),
        tile("arrows", str(len(aligned.arrows))),
        tile("waiting", "n/a" if waiting < 0 else f"{waiting:.0f}%"),
    ])
    run_id = escape(str(run.metadata.get("run_id", "unknown")))
    return (
        f"<h1>{escape(title)}</h1>\n"
        f'<p class="sub mono">run {run_id}</p>\n'
        f'<div class="tiles">{tiles}</div>'
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
        sections.append(
            f"<h3>{escape(lane)}"
            f"{f' <span class=\"note\">{escape(role)}</span>' if role else ''}</h3>\n"
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


def _sequence_row(span: PlacedSpan, origin: int) -> str:
    """One call in a lane's sequence, indented to show what it was called from."""
    indent = "&nbsp;" * (4 * min(span.depth, _MAX_DEPTH_DRAWN))
    arrow = "↳ " if span.depth else ""
    wait = "n/a" if span.wait_pct < 0 else f"{span.wait_pct:.0f}%"
    return (
        f'<tr><td class="mono">{span.depth}</td>'
        f'<td class="mono">{indent}{arrow}{escape(span.name)}</td>'
        f"<td>{escape(format_ns(span.t0_ns - origin))}</td>"
        f"<td>{escape(format_ns(span.duration_ns))}</td>"
        f"<td>{wait}</td></tr>"
    )


def _canvas_block() -> str:
    """The timeline itself, drawn by the inlined script into a canvas."""
    return (
        "<h2>Timeline</h2>\n"
        '<div class="tl-controls">'
        '<button type="button" id="tl-reset">reset zoom</button>'
        '<button type="button" id="tl-critical">show critical path</button>'
        '<span class="note" id="tl-range"></span>'
        "</div>\n"
        '<div id="tl-wrap"><canvas id="tl-canvas"></canvas>'
        '<div id="tl-tip" hidden></div></div>\n'
        '<p class="note">Drag to pan, scroll to zoom, hover for detail, click a span to '
        "trace what it was waiting for. Width is wall time; colour is the share of it spent "
        "waiting rather than running.</p>"
    )


def _critical_path_block(chain: list[PlacedSpan], aligned: AlignedTrace) -> str:
    """The chain that actually set the run's duration, newest first.

    The single most useful thing on the page: it converts "everything looks a bit slow" into
    an ordered list of what was waiting on what.
    """
    if not chain:
        return ""
    rows = []
    for index, span in enumerate(chain):
        following = chain[index - 1] if index else None
        gap = ""
        if following is not None:
            idle = following.t0_ns - span.t1_ns
            if idle > 0:
                gap = f"{format_ns(idle)} idle after"
        rows.append(
            f"<tr><td>{escape(span.worker)}</td>"
            f"<td>{escape('/'.join(span.path))}</td>"
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


def _caveats(aligned: AlignedTrace) -> str:
    """Everything that makes this timeline less than the whole truth.

    A dropped span, an unmatched wait and a cross-host clock are all reasons a reader should
    trust the picture slightly less, and each is invisible unless stated.
    """
    items: list[str] = []
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
    unmeasured = sum(1 for span in aligned.spans if not span.cpu_measured)
    if unmeasured:
        items.append(
            f"{unmeasured:,} span(s) were derived from function calls, which cannot measure "
            "CPU time; they are drawn hatched and their wait is unknown, not zero.",
        )
    items.append(alignment_accuracy_note(aligned.hosts))
    entries = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<h2>Caveats</h2>\n<ul>{entries}</ul>"


def _payload(
    aligned: AlignedTrace,
    chain: list[PlacedSpan],
    run: MergedRun,
) -> JsonValue:
    """The data the page draws from, also embedded for anything that wants to read it.

    Times are re-based to the trace's own start and expressed in microseconds: the absolute
    epoch nanosecond is both unreadable and beyond a JavaScript integer, and the offset is
    what every figure on the page is actually about.
    """
    origin = aligned.t0_ns
    drawn, omitted = _spans_to_draw(aligned.spans)
    lane_index = {lane: index for index, lane in enumerate(aligned.lanes)}
    critical = {id(span) for span in chain}

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
            }
            for span in drawn
        ],
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


def _spans_to_draw(spans: list[PlacedSpan]) -> tuple[list[PlacedSpan], int]:
    """Cap what the page carries, keeping the longest spans and counting the rest.

    The longest rather than the first: a reader zooming into a busy region wants the shape of
    it, and the spans that define that shape are the ones with visible width. The number
    dropped is reported, never hidden.
    """
    if len(spans) <= _MAX_SPANS_DRAWN:
        return spans, 0
    kept = sorted(spans, key=lambda span: -span.duration_ns)[:_MAX_SPANS_DRAWN]
    kept.sort(key=lambda span: span.t0_ns)
    return kept, len(spans) - _MAX_SPANS_DRAWN


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
  var gpu = data.gpu || [];
  if (!spans.length) { return; }

  var ctx = el.getContext('2d');
  var tip = document.getElementById('tl-tip');
  var wrap = document.getElementById('tl-wrap');
  var rangeLabel = document.getElementById('tl-range');
  var ROW_H = 15, LANE_PAD = 8, PAD_L = 132, PAD_T = 26, GPU_H = 34;
  var total = data.duration_us || 1;
  var view = { t0: 0, t1: total };
  var onlyCritical = false, hover = null, focus = null;

  // Each lane is as tall as it is deep: one row per nesting level, so a callee is drawn
  // under its caller instead of over it. Computed once — every y on the page reads it.
  var laneTop = [], laneH = [], lanesBottom = PAD_T;
  for (var li = 0; li < lanes.length; li++) {
    var rows = lanes[li].rows || 1;
    laneTop[li] = lanesBottom;
    laneH[li] = rows * ROW_H + LANE_PAD;
    lanesBottom += laneH[li];
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
    rangeLabel.textContent = fmt(view.t1 - view.t0) + ' shown of ' + fmt(total);
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
    var label = lanes[i].id;
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
    var x0 = xOf(sp.t, width), x1 = xOf(sp.t + sp.d, width);
    if (x1 < PAD_L || x0 > width) { return; }
    x0 = Math.max(x0, PAD_L);
    var w = Math.max(x1 - x0, 1);
    var y = laneTop[sp.l] + (sp.y || 0) * ROW_H + 3, h = ROW_H - 3;
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
    if (w > 34) {
      var name = sp.n.split('/').pop();
      if (name.length * 6 < w - 6) {
        ctx.fillStyle = '#ffffff';
        ctx.fillText(name, x0 + 4, y + h / 2);
      }
    }
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
      if (sp.l !== lane || (sp.y || 0) !== row) { continue; }
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
    if (sp.c) { tip.appendChild(named('div', 'crit', 'on the critical path')); }
    tip.hidden = false;
    var x = event.clientX - box.left + 14, y = event.clientY - box.top + 14;
    if (x + tip.offsetWidth > box.width) { x = box.width - tip.offsetWidth - 6; }
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });
  el.addEventListener('mouseleave', function () { tip.hidden = true; hover = null; });
  el.addEventListener('click', function () { focus = (focus === hover) ? null : hover; draw(); });

  var dragging = false, dragX = 0, dragT = 0;
  el.addEventListener('mousedown', function (event) {
    dragging = true; dragX = event.clientX; dragT = view.t0;
    el.classList.add('dragging'); tip.hidden = true;
  });
  window.addEventListener('mouseup', function () {
    dragging = false; el.classList.remove('dragging');
  });
  window.addEventListener('mousemove', function (event) {
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
    view.t0 = 0; view.t1 = total; focus = null; draw();
  });
  var criticalButton = document.getElementById('tl-critical');
  criticalButton.addEventListener('click', function () {
    onlyCritical = !onlyCritical;
    criticalButton.setAttribute('aria-pressed', onlyCritical ? 'true' : 'false');
    draw();
  });

  window.addEventListener('resize', resize);
  resize();
})();
"""
