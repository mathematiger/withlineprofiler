"""Offline derivation over recorded traces: one clock, matched links, the critical path.

Everything here is pure: it reads what the workers recorded and derives from it, the same way
:mod:`lineprofiler.accounting.analysis` derives from samples. No I/O, no clocks of its own —
so it is fully testable from fixed inputs, which matters because the arithmetic below is
exactly where a plausible-looking wrong answer would hide.

The three jobs, in order:

1. **One axis.** ``perf_counter_ns`` has an arbitrary per-process origin, so raw span
   timestamps from two workers are not comparable. Each worker's clock anchors tie its
   monotonic clock to the wall clock, which every process shares.
2. **Arrows.** A ``wait_on`` is matched to the most recent ``signal`` of the same
   channel and key. Unmatched waits are *reported*, never dropped.
3. **The critical path.** Walking backwards from the last span through wait→signal edges
   gives the chain that actually determined how long the run took — which is the difference
   between "this worker was idle a lot" and "this worker was idle *because of that one*".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from lineprofiler.accounting.trace import FLAG_AUTO, ClockAnchor, Link, Origin, WorkerTrace

NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class PlacedSpan:
    """A span mapped onto the common epoch, with its worker and phase resolved.

    ``t0_ns``/``t1_ns`` are nanoseconds since the Unix epoch, so spans from any worker on any
    host can be compared directly — within the accuracy caveat that
    :func:`alignment_accuracy_note` spells out.
    """

    worker: str
    role: str
    thread_id: int
    path: tuple[str, ...]
    t0_ns: int
    t1_ns: int
    cpu_ns: int
    flags: int
    origin: Origin | None = None
    """Where the code behind this span is defined, or ``None`` when it was not recorded.

    Present for spans derived from function calls, absent for a phase the user named — there
    is no code object behind a name. Callers must handle the absence rather than substitute a
    placeholder file, which would point a reader at source that has nothing to do with it.
    """
    depth: int = 0
    """How many spans enclose this one on its lane: 0 is a top-level call, 1 its callee.

    This is what makes the lane readable as a call structure rather than a list. It is derived
    offline in :func:`place_spans`, never recorded — the phase path already *is* the call
    stack, so for a named phase the depth costs nothing to know.
    """

    @property
    def duration_ns(self) -> int:
        """Wall time this span covered on the common axis."""
        return max(0, self.t1_ns - self.t0_ns)

    @property
    def cpu_measured(self) -> bool:
        """Whether CPU time was measured for this span, so its wait is meaningful."""
        return self.cpu_ns >= 0

    @property
    def wait_ns(self) -> int:
        """Wall time not on a CPU. Only meaningful when :attr:`cpu_measured`."""
        if not self.cpu_measured:
            return 0
        return max(0, self.duration_ns - self.cpu_ns)

    @property
    def wait_pct(self) -> float:
        """Share of this span spent waiting, ``-1.0`` when CPU time was not measured.

        Negative rather than zero, so a renderer cannot accidentally shade an unmeasured span
        as fully busy. Callers test the sign.
        """
        if not self.cpu_measured or self.duration_ns <= 0:
            return -1.0 if not self.cpu_measured else 0.0
        return 100.0 * self.wait_ns / self.duration_ns

    @property
    def lane(self) -> str:
        """Identity of the row this span is drawn on: one lane per worker thread."""
        return f"{self.worker}#{self.thread_id}"

    @property
    def name(self) -> str:
        """The phase's own name, without its ancestors."""
        return self.path[-1] if self.path else ""


@dataclass(frozen=True, slots=True)
class Arrow:
    """A matched dependency: ``src`` produced what ``dst`` was waiting for."""

    channel: str
    key: str
    src_worker: str
    dst_worker: str
    src_t_ns: int
    dst_t_ns: int

    @property
    def delay_ns(self) -> int:
        """How long the waiter sat after the signal — its share of the blocked time.

        Negative would mean the waiter was released before the producer signalled, which the
        matcher never emits: it is the clock-skew symptom :func:`match_links` clamps.
        """
        return max(0, self.dst_t_ns - self.src_t_ns)


@dataclass(slots=True)
class AlignedTrace:
    """Every worker's spans on one axis, plus the arrows and what could not be matched."""

    spans: list[PlacedSpan] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)
    unmatched_waits: list[tuple[str, str, str]] = field(default_factory=list)
    """``(worker, channel, key)`` for each ``wait_on`` with no matching ``signal``."""
    dropped_spans: int = 0
    dropped_links: int = 0
    lanes: list[str] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    """Lane identity to the role of the worker that owns it."""
    hosts: set[str] = field(default_factory=set)
    lifecycle_marks: list[Link] = field(default_factory=list)
    """Request-lifecycle checkpoints, with ``t_ns`` already on the common epoch.

    Kept apart from ``arrows`` because they answer a different question: an arrow says who
    released whom, while these decompose one request's wait into named segments stamped in
    the processes that own each transition."""
    clock_steps: dict[str, int] = field(default_factory=dict)
    """Lane-owning worker to the number of clock anchors :func:`usable_anchors` rejected.

    A non-empty entry means that worker's wall clock moved mid-run, so its spans are placed
    by monotonic offset from the run's first anchor rather than by the fit the later ones
    imply. Durations are unaffected — they come from ``perf_counter`` — but absolute
    placement after the step, and therefore alignment against another worker, is only as
    good as the origin. Repairing the axis without saying so would leave a reader trusting a
    cross-lane gap this module cannot vouch for."""

    @property
    def t0_ns(self) -> int:
        """Start of the earliest span, or ``0`` when there are none."""
        return min((span.t0_ns for span in self.spans), default=0)

    @property
    def t1_ns(self) -> int:
        """End of the latest span, or ``0`` when there are none."""
        return max((span.t1_ns for span in self.spans), default=0)

    @property
    def duration_ns(self) -> int:
        """Wall time the whole trace covers."""
        return max(0, self.t1_ns - self.t0_ns)

    @property
    def is_complete(self) -> bool:
        """Whether every recorded span survived — a wrapped ring means it did not."""
        return self.dropped_spans == 0 and self.dropped_links == 0


_CLOCK_STEP_FLOOR_NS = 1_000_000
"""How far the two clocks may disagree over an interval too short to judge by ratio.

Anchors taken within the same millisecond carry no usable elapsed time to compare against, so
the ratio test degenerates. A millisecond floor keeps that case decidable without rejecting
two readings taken back to back."""


def usable_anchors(anchors: list[ClockAnchor]) -> list[ClockAnchor]:
    """Drop the anchors whose wall-clock reading cannot be reconciled with the monotonic one.

    ``perf_counter_ns`` is monotonic and steady; ``time_ns`` is neither. An NTP step, a
    resumed VM or a container clock correction moves the wall clock by seconds to hours
    between two anchors, and :func:`to_common_epoch` would fit a line straight through the
    pair — mapping every span in that bracket onto an axis dilated by four or five orders of
    magnitude, and *reversed* when the step went backwards. Measured on a real 0.25 s run
    with one backward hour step: a 40 ms interval placed as −1,440 s, every span after the
    step drawn as ``0ns``, the page's headline reading ``57m 48s`` and its top finding
    claiming the lane had no phase open for 100% of the run.

    An anchor is kept when the wall time elapsed since the first anchor is within a factor of
    two of the monotonic time elapsed. That band is far wider than any drift the anchors
    exist to correct — slew is parts per million, and even chrony's aggressive default caps
    at ~8% — and far narrower than any step worth catching, so this rejects steps without
    rejecting the correction it would be pointless to keep anchors for.

    The first anchor is the origin and is always kept: :meth:`Profiler._start_tracing` takes
    it to date the worker's monotonic clock, before anything could have moved. Where the whole
    remaining run disagrees with it, one of the two readings is right and nothing here can
    know which — so the mapping stays consistent with the run's own start and
    :attr:`AlignedTrace.clock_steps` reports that it had to.
    """
    if len(anchors) < 2:
        return list(anchors)
    ordered = sorted(anchors, key=lambda anchor: anchor.perf_ns)
    origin = ordered[0]
    kept = [origin]
    for anchor in ordered[1:]:
        elapsed = anchor.perf_ns - origin.perf_ns
        disagreement = abs((anchor.real_ns - origin.real_ns) - elapsed)
        if disagreement <= elapsed + _CLOCK_STEP_FLOOR_NS:
            kept.append(anchor)
    return kept


def to_common_epoch(perf_ns: int, anchors: list[ClockAnchor]) -> int:
    """Map one worker-local ``perf_counter_ns`` reading onto the Unix epoch.

    With several anchors the two nearest bracketing ones give a local linear fit, which
    absorbs drift between the monotonic and wall clocks over a long run. With one anchor the
    fit degenerates to a constant offset, which is correct at the anchor and drifts slowly
    away from it — at typical rates, well under the NTP error that already bounds cross-host
    accuracy.

    With no anchors at all the reading is returned unchanged. That is not a valid epoch time,
    but it is the only honest answer: there is nothing to map through, and inventing an
    offset would place the lane confidently in the wrong place.
    """
    if not anchors:
        return perf_ns
    if len(anchors) == 1:
        only = anchors[0]
        return only.real_ns + (perf_ns - only.perf_ns)

    ordered = sorted(anchors, key=lambda anchor: anchor.perf_ns)
    before, after = _bracket(ordered, perf_ns)
    span = after.perf_ns - before.perf_ns
    if span <= 0:
        return before.real_ns + (perf_ns - before.perf_ns)
    ratio = (after.real_ns - before.real_ns) / span
    return int(before.real_ns + (perf_ns - before.perf_ns) * ratio)


def _bracket(
    ordered: list[ClockAnchor],
    perf_ns: int,
) -> tuple[ClockAnchor, ClockAnchor]:
    """Return the two anchors bracketing ``perf_ns``, or the nearest pair when outside.

    Extrapolating from the nearest pair rather than clamping keeps a span recorded just
    before the first anchor or just after the last one in the right order relative to its
    neighbours, which clamping would collapse.
    """
    for index in range(len(ordered) - 1):
        if ordered[index].perf_ns <= perf_ns <= ordered[index + 1].perf_ns:
            return ordered[index], ordered[index + 1]
    if perf_ns < ordered[0].perf_ns:
        return ordered[0], ordered[1]
    return ordered[-2], ordered[-1]


def place_spans(
    trace: WorkerTrace,
    worker: str,
    role: str,
) -> list[PlacedSpan]:
    """Map one worker's spans onto the common epoch, resolving their phase paths and depth."""
    anchors = usable_anchors(trace.anchors)
    placed = []
    for span in trace.spans:
        path = tuple(trace.path_of(span.phase_id))
        placed.append(
            PlacedSpan(
                worker=worker,
                role=role,
                thread_id=span.thread_id,
                path=path,
                t0_ns=to_common_epoch(span.t0_ns, anchors),
                t1_ns=to_common_epoch(span.t1_ns, anchors),
                cpu_ns=span.cpu_ns,
                flags=span.flags,
                origin=trace.origin_of(span.phase_id),
                depth=max(0, len(path) - 1),
            ),
        )
    return _with_derived_depth(placed)


def _with_derived_depth(spans: list[PlacedSpan]) -> list[PlacedSpan]:
    """Fill in the depth of spans whose phase path does not carry their ancestry.

    A named phase's path *is* its call stack, so its depth came for free above. An
    auto-derived span's path is the function's qualname — a name, not an ancestry — so every
    one of them would otherwise sit at depth 0 and a whole auto-traced lane would draw as a
    single overpainted row. Those are recovered by containment instead, per thread, which is
    the ordinary flame-graph construction.
    """
    auto = [span for span in spans if span.flags & FLAG_AUTO]
    if not auto:
        return spans

    depths: dict[int, int] = {}
    by_thread: dict[int, list[PlacedSpan]] = {}
    for span in auto:
        by_thread.setdefault(span.thread_id, []).append(span)

    for thread_spans in by_thread.values():
        # Longest-first at equal starts, so a parent is always seen before its children.
        thread_spans.sort(key=lambda s: (s.t0_ns, -s.t1_ns))
        stack: list[PlacedSpan] = []
        for span in thread_spans:
            while stack and stack[-1].t1_ns <= span.t0_ns:
                stack.pop()
            depths[id(span)] = len(stack)
            stack.append(span)

    return [
        replace(span, depth=depths[id(span)]) if id(span) in depths else span
        for span in spans
    ]


def max_depth_of(trace: AlignedTrace, lane: str) -> int:
    """Deepest nesting level reached on ``lane``, so a renderer can size the row for it."""
    return max((span.depth for span in trace.spans if span.lane == lane), default=0)


def match_links(
    links_by_worker: dict[str, tuple[list[Link], list[ClockAnchor]]],
) -> tuple[list[Arrow], list[tuple[str, str, str]]]:
    """Join each ``wait_on`` to the most recent ``signal`` of the same channel and key.

    Matching is *global* across workers, which is the point: the producer and the consumer are
    different processes. The most recent preceding signal wins, so a channel reused every
    iteration links each wait to that iteration's producer rather than to the first one.

    A wait with no signal is returned separately rather than dropped or raised. Half of a
    pipeline being instrumented is the normal state of an incremental rollout, and it must
    degrade to fewer arrows — never to a crash, and never to a silently missing dependency.
    """
    signals: list[tuple[int, str, str, str]] = []
    waits: list[tuple[int, str, str, str]] = []
    for worker, (links, anchors) in links_by_worker.items():
        for link in links:
            placed = to_common_epoch(link.t_ns, anchors)
            entry = (placed, worker, link.channel, link.key)
            if link.kind == "signal":
                signals.append(entry)
            elif link.kind == "wait":
                waits.append(entry)

    by_channel: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for placed, worker, channel, key in signals:
        by_channel.setdefault((channel, key), []).append((placed, worker))
    for entries in by_channel.values():
        entries.sort()

    arrows: list[Arrow] = []
    unmatched: list[tuple[str, str, str]] = []
    for placed, worker, channel, key in sorted(waits):
        candidates = by_channel.get((channel, key))
        source = _latest_before(candidates, placed) if candidates else None
        if source is None:
            unmatched.append((worker, channel, key))
            continue
        src_t, src_worker = source
        arrows.append(
            Arrow(
                channel=channel,
                key=key,
                src_worker=src_worker,
                dst_worker=worker,
                src_t_ns=src_t,
                dst_t_ns=placed,
            ),
        )
    return arrows, unmatched


def _latest_before(
    candidates: list[tuple[int, str]],
    when: int,
) -> tuple[int, str] | None:
    """The latest signal at or before ``when``, else the earliest signal overall.

    Falling back to the earliest rather than giving up handles the ordinary skew case: on two
    hosts a signal can be timestamped a few hundred microseconds *after* the wait it released.
    The dependency is real and worth drawing; :attr:`Arrow.delay_ns` clamps the negative
    interval so the arrow never claims the waiter was released before the producer ran.
    """
    best: tuple[int, str] | None = None
    for entry in candidates:
        if entry[0] <= when:
            best = entry
        else:
            break
    return best if best is not None else candidates[0]


def critical_path(trace: AlignedTrace) -> list[PlacedSpan]:
    """The chain of spans that determined the run's length, latest first.

    Walks backwards from the last span to finish. At each step it stays on the same lane
    while spans abut, and hops to another worker when an arrow explains the gap — so the
    result reads as "this finished late because it waited for that, which waited for this".

    This is the answer to "where does the waiting time come from" that a lane view alone
    cannot give: a chart shows *that* a worker idled, the critical path shows *whose* work it
    was idling on.
    """
    if not trace.spans:
        return []

    by_lane: dict[str, list[PlacedSpan]] = {}
    for span in trace.spans:
        by_lane.setdefault(span.lane, []).append(span)
    for spans in by_lane.values():
        spans.sort(key=lambda s: s.t0_ns)

    arrivals = _arrivals_by_worker(trace.arrows)
    current: PlacedSpan | None = max(trace.spans, key=lambda s: s.t1_ns)
    chain: list[PlacedSpan] = []
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = _predecessor(current, by_lane, arrivals, seen)
    return chain


def _arrivals_by_worker(arrows: list[Arrow]) -> dict[str, list[Arrow]]:
    """Arrows grouped by the worker that was waiting, earliest first."""
    grouped: dict[str, list[Arrow]] = {}
    for arrow in arrows:
        grouped.setdefault(arrow.dst_worker, []).append(arrow)
    for entries in grouped.values():
        entries.sort(key=lambda a: a.dst_t_ns)
    return grouped


def _predecessor(
    span: PlacedSpan,
    by_lane: dict[str, list[PlacedSpan]],
    arrivals: dict[str, list[Arrow]],
    seen: set[int],
) -> PlacedSpan | None:
    """What this span was waiting on: the arrow that released it, else its lane predecessor.

    The arrow is preferred when one lands inside the span, because a cross-process dependency
    explains a gap that the same lane's previous span cannot.
    """
    arrow = _releasing_arrow(span, arrivals)
    if arrow is not None:
        producer = _span_covering(by_lane, arrow.src_worker, arrow.src_t_ns, seen)
        if producer is not None:
            return producer

    lane = by_lane.get(span.lane, [])
    previous = None
    for candidate in lane:
        if candidate.t1_ns <= span.t0_ns and id(candidate) not in seen:
            previous = candidate
        elif candidate.t0_ns >= span.t1_ns:
            break
    return previous


_ARRIVAL_SLACK_NS = 2_000_000
"""How long after a span ends a ``wait_on`` may still refer to it.

The documented idiom puts the call *after* the blocking get returns::

    with profiler.phase("queue_get"):
        batch = queue.get()
    profiler.wait_on("batch", batch.id)

so its timestamp lands just past the span's end — measured at 10–70 µs on an ordinary
pipeline, and more when the release is followed by deserialisation. Requiring the arrow to
fall strictly inside the span therefore matched nothing at all, and the critical path never
left the blocked worker's own lane: it reported *that* the learner waited without ever
naming who it waited for, which is the one thing it exists to do.

Two milliseconds is far longer than the gap needs to be and far shorter than the waits worth
explaining, so it cannot bridge to the wrong span."""


def _releasing_arrow(span: PlacedSpan, arrivals: dict[str, list[Arrow]]) -> Arrow | None:
    """The arrow whose wait this span was blocked on, if any.

    Accepts an arrival shortly *after* the span ends as well as one inside it — see
    :data:`_ARRIVAL_SLACK_NS`. Where several qualify, the latest wins: it is the one whose
    release actually ended this wait, the earlier ones having been consumed by earlier spans.
    """
    best: Arrow | None = None
    for arrow in arrivals.get(span.worker, []):
        if span.t0_ns <= arrow.dst_t_ns <= span.t1_ns + _ARRIVAL_SLACK_NS:
            best = arrow
    return best


def _span_covering(
    by_lane: dict[str, list[PlacedSpan]],
    worker: str,
    when: int,
    seen: set[int],
) -> PlacedSpan | None:
    """The span on ``worker`` that was running at ``when``, else the last one before it."""
    best: PlacedSpan | None = None
    for lane, spans in by_lane.items():
        if not lane.startswith(f"{worker}#"):
            continue
        for span in spans:
            if id(span) in seen:
                continue
            if span.t0_ns <= when <= span.t1_ns:
                return span
            if span.t1_ns <= when and (best is None or span.t1_ns > best.t1_ns):
                best = span
    return best


def lane_busy_share(trace: AlignedTrace, lane: str) -> float:
    """Share of the trace's span of time during which ``lane`` had a phase open.

    "Open" is not "working": a lane blocked in ``queue_get`` for the whole run scores 100%
    here, because a phase was indeed open the whole time. Pair this with
    :func:`lane_working_share` — the gap between the two *is* the waiting the timeline exists
    to explain. Overlapping spans on one lane (a phase inside a phase) are counted once, so
    nesting cannot push this above 100%.
    """
    return _covered_share(trace, lane, only_working=False)


def lane_working_share(trace: AlignedTrace, lane: str) -> float:
    """Share of the trace during which ``lane`` was on a CPU, ``-1.0`` when unmeasured.

    Where :func:`lane_busy_share` counts a blocked ``queue_get`` as busy, this counts only
    the CPU time inside each span, so an idle worker reads as idle. Returns ``-1.0`` when no
    span on the lane measured CPU time — auto-derived spans do not — because a lane whose
    work is unknown must not be drawn as a lane that did none.
    """
    spans = [span for span in trace.spans if span.lane == lane]
    if not spans or not any(span.cpu_measured for span in spans):
        return -1.0
    if trace.duration_ns <= 0:
        return 0.0
    cpu = sum(span.cpu_ns for span in spans if span.cpu_measured and _is_leaf(span, spans))
    return min(100.0, 100.0 * cpu / trace.duration_ns)


@dataclass(frozen=True, slots=True)
class Segment:
    """One named interval of a request's life, aggregated over every request on a channel.

    ``total_ns`` sums the segment across requests, so the shares compare like with like: the
    question is which part of the wait dominates, and one request's breakdown answers it only
    by accident.
    """

    channel: str
    name: str
    total_ns: int
    count: int

    @property
    def mean_ns(self) -> float:
        """Average length of this segment across the requests that reported it."""
        return self.total_ns / self.count if self.count else 0.0


def lifecycle_segments(trace: AlignedTrace) -> dict[str, list[Segment]]:
    """Decompose each channel's request lifecycles into consecutive named segments.

    Returns segments per channel, in the order the checkpoints occurred, so the result reads
    as a breakdown of one request's journey rather than a set of unrelated totals.

    Only complete, monotonic lifecycles contribute. A request whose checkpoints arrived out of
    order across processes — the ordinary consequence of clock skew between hosts — is
    dropped rather than contributing a negative segment, because a segment that cannot have
    happened is worse than one that is missing.

    Test specifically:
        - a server sleeping a known 50 ms before admitting and 20 ms computing attributes
          ≈50 ms and ≈20 ms to the right segments
        - a lifecycle missing its end contributes nothing
        - checkpoints arriving out of order are dropped, never counted as negative
    """
    by_request: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for link in trace.lifecycle_marks:
        by_request.setdefault((link.channel, link.key), []).append((link.t_ns, link.kind))

    totals: dict[tuple[str, str], list[int]] = {}
    order: dict[str, list[str]] = {}
    for (channel, _key), marks in by_request.items():
        for name, span_ns in _segments_of(marks):
            bucket = totals.setdefault((channel, name), [0, 0])
            bucket[0] += span_ns
            bucket[1] += 1
            names = order.setdefault(channel, [])
            if name not in names:
                names.append(name)

    segments: dict[str, list[Segment]] = {}
    for channel, names in order.items():
        segments[channel] = [
            Segment(
                channel=channel,
                name=name,
                total_ns=totals[(channel, name)][0],
                count=totals[(channel, name)][1],
            )
            for name in names
        ]
    return segments


def _segments_of(marks: list[tuple[int, str]]) -> list[tuple[str, int]]:
    """Turn one request's checkpoints into consecutive ``from → to`` intervals.

    Requires both ends of the lifecycle: a request still in flight when the trace was cut has
    a genuine but unknown remainder, and closing it at the last mark would invent a fast
    request out of an unfinished one.
    """
    kinds = {kind for _at, kind in marks}
    if "begin" not in kinds or "end" not in kinds:
        return []
    ordered = sorted(marks)
    if ordered[0][1] != "begin" or ordered[-1][1] != "end":
        return []  # skewed or duplicated: not a lifecycle we can read

    segments = []
    for (start, from_kind), (end, to_kind) in zip(ordered, ordered[1:], strict=False):
        if end < start:
            return []
        segments.append((f"{_label_of(from_kind)} → {_label_of(to_kind)}", end - start))
    return segments


def _label_of(kind: str) -> str:
    """Human name for a lifecycle checkpoint: ``mark:admitted`` reads as ``admitted``."""
    return kind[5:] if kind.startswith("mark:") else kind


def overlap_ns(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
) -> int:
    """Nanoseconds covered by both interval lists — their intersection, not their union.

    The primitive behind "is this a hang, or is it queueing behind real work?", which is the
    first question anyone asks about a phase that is 96% wait and has nothing in common with
    the other answer as a problem. Answering it previously meant extracting the JSON embedded
    in an 11.7 MB page and intersecting the intervals by hand.

    Overlaps *within* each list are merged first, so a nested phase cannot make its own lane
    count twice and push the result past either input's extent.

    Test specifically:
        - disjoint inputs overlap by zero
        - identical inputs overlap by their own length
        - nested intervals on one side are counted once, not once per level
        - the result is symmetric in its arguments
    """
    left = _merged(first)
    right = _merged(second)
    total = 0
    index = 0
    for start, end in left:
        # Advance past everything that ends before this interval opens; both lists are sorted,
        # so the cursor never rewinds and the walk stays linear rather than quadratic.
        while index < len(right) and right[index][1] <= start:
            index += 1
        probe = index
        while probe < len(right) and right[probe][0] < end:
            total += min(end, right[probe][1]) - max(start, right[probe][0])
            probe += 1
    return total


def _merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and coalesce overlapping intervals, dropping empty ones."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def concurrent_activity(
    trace: AlignedTrace,
    blocked: list[PlacedSpan],
) -> dict[str, float]:
    """Share of ``blocked``'s total time during which each *other* role had a phase open.

    Turns "this worker waited 280 s" into "and the server was computing for 88% of it", which
    is the difference between a hang and a queue — two problems with nothing in common that
    look identical in a phase table.

    Returns a share per role in ``0.0..100.0``, excluding the lanes the blocked spans are
    themselves on: a lane cannot be concurrent with itself in any useful sense.

    Test specifically:
        - a worker blocked 100 ms while another is busy exactly 60 ms reports ≈60%
        - a role on several lanes is counted once, not once per lane
        - blocked spans with no concurrent activity report nothing
    """
    windows = _merged([(span.t0_ns, span.t1_ns) for span in blocked])
    total = sum(end - start for start, end in windows)
    if total <= 0:
        return {}

    own = {span.lane for span in blocked}
    by_role: dict[str, list[tuple[int, int]]] = {}
    for span in trace.spans:
        if span.lane in own:
            continue
        by_role.setdefault(span.role or span.lane, []).append((span.t0_ns, span.t1_ns))

    # Grouped by role rather than lane: "the inference server was busy" is the finding, and
    # spelling it as four lane ids — each a worker#thread that means nothing to the reader —
    # both buries it and understates it, since any one thread covers a fraction of what the
    # role as a whole covered. Merging inside `overlap_ns` keeps the union honest.
    return {
        role: 100.0 * overlap_ns(windows, intervals) / total
        for role, intervals in by_role.items()
    }


def role_occupancy(trace: AlignedTrace) -> dict[str, tuple[float, float]]:
    """Mean ``(busy, working)`` share per role, over that role's lanes.

    The pair is the bottleneck statement for a queue-driven worker: "busy 97%, working 35%"
    says the process was inside a phase almost always and on a CPU rarely, which is a
    different problem from either number alone. Averaged over lanes rather than summed, so
    the figures stay comparable to the per-lane ones a reader sees on the timeline.

    Lanes whose CPU time was never measured are excluded from the working mean rather than
    counted as zero; a role with no measured lane reports ``-1.0`` for it, the convention this
    module uses everywhere for "not measured".
    """
    busy_by_role: dict[str, list[float]] = {}
    working_by_role: dict[str, list[float]] = {}
    for lane in trace.lanes:
        role = trace.roles.get(lane, "")
        busy_by_role.setdefault(role, []).append(lane_busy_share(trace, lane))
        working = lane_working_share(trace, lane)
        if working >= 0:
            working_by_role.setdefault(role, []).append(working)

    occupancy: dict[str, tuple[float, float]] = {}
    for role, busy_shares in busy_by_role.items():
        working_shares = working_by_role.get(role, [])
        occupancy[role] = (
            sum(busy_shares) / len(busy_shares),
            sum(working_shares) / len(working_shares) if working_shares else -1.0,
        )
    return occupancy


def _is_leaf(span: PlacedSpan, siblings: list[PlacedSpan]) -> bool:
    """Whether no other span on the lane sits strictly inside ``span``.

    CPU time is cumulative through nesting — a parent's ``cpu_ns`` already includes its
    children's — so summing every span would count the same microseconds once per level.
    Counting only the innermost spans totals each one exactly once.
    """
    return not any(
        other is not span
        and other.t0_ns >= span.t0_ns
        and other.t1_ns <= span.t1_ns
        and other.duration_ns < span.duration_ns
        for other in siblings
    )


def _covered_share(trace: AlignedTrace, lane: str, only_working: bool) -> float:
    """Share of the trace covered by ``lane``'s spans, merging overlaps."""
    if trace.duration_ns <= 0:
        return 0.0
    intervals = sorted(
        (span.t0_ns, span.t1_ns)
        for span in trace.spans
        if span.lane == lane and (not only_working or span.cpu_measured)
    )
    covered = 0
    cursor = 0
    for start, end in intervals:
        start = max(start, cursor)
        if end > start:
            covered += end - start
            cursor = end
    return 100.0 * covered / trace.duration_ns


def align_run(run: object) -> AlignedTrace:
    """Build the aligned trace for a merged run whose workers carry trace data.

    Takes the :class:`~lineprofiler.accounting.snapshot.MergedRun` structurally rather than by
    import, keeping this module free of the dependency and testable from plain objects.
    """
    workers = list(getattr(run, "workers", []))
    aligned = AlignedTrace()
    links_by_worker: dict[str, tuple[list[Link], list[ClockAnchor]]] = {}

    labels = _distinct_labels(workers)
    for worker in workers:
        trace = getattr(worker, "trace", None)
        if trace is None or (not trace.spans and not trace.links):
            continue
        label = labels[id(worker)]
        role = str(getattr(worker, "role", "main"))
        anchors = usable_anchors(trace.anchors)
        rejected = len(trace.anchors) - len(anchors)
        if rejected:
            aligned.clock_steps[label] = rejected
        aligned.spans.extend(place_spans(trace, label, role))
        links_by_worker[label] = (trace.links, anchors)
        aligned.dropped_spans += trace.dropped
        aligned.dropped_links += trace.dropped_links
        host = getattr(worker, "host", None)
        if host:
            aligned.hosts.add(str(host))

    aligned.arrows, aligned.unmatched_waits = match_links(links_by_worker)
    aligned.lifecycle_marks = _placed_lifecycle_marks(links_by_worker)
    aligned.spans.sort(key=lambda span: span.t0_ns)
    aligned.lanes = _ordered_lanes(aligned.spans)
    aligned.roles = {span.lane: span.role for span in aligned.spans}
    return aligned


def _placed_lifecycle_marks(
    links_by_worker: dict[str, tuple[list[Link], list[ClockAnchor]]],
) -> list[Link]:
    """Lifecycle checkpoints from every worker, moved onto the common epoch.

    The alignment is what makes the decomposition possible at all: each checkpoint is stamped
    by whichever process owns that transition, on its own ``perf_counter`` origin, so they are
    only comparable once mapped through that worker's anchors.
    """
    placed: list[Link] = []
    for links, anchors in links_by_worker.values():
        for link in links:
            if link.kind in {"signal", "wait"}:
                continue
            placed.append(
                Link(
                    channel=link.channel,
                    key=link.key,
                    kind=link.kind,
                    t_ns=to_common_epoch(link.t_ns, anchors),
                    thread_id=link.thread_id,
                ),
            )
    return placed


def _distinct_labels(workers: list[object]) -> dict[int, str]:
    """Give every worker a lane name no other worker shares.

    ``WorkerSnapshot.label`` prefers the rank a launcher assigned, which is right for a
    report: rank is what the reader recognises. A timeline needs more, because rank is only
    unique when a launcher actually assigned one — four processes started by
    ``multiprocessing`` all inherit the same rank (often 0), and collapsing them onto one
    lane hides the very interleaving the page exists to show. The pid disambiguates, and is
    appended only where it is needed so the common case stays readable.
    """
    counts: dict[str, int] = {}
    for worker in workers:
        base = str(getattr(worker, "label", "worker"))
        counts[base] = counts.get(base, 0) + 1

    labels: dict[int, str] = {}
    for worker in workers:
        base = str(getattr(worker, "label", "worker"))
        role = str(getattr(worker, "role", ""))
        if counts[base] > 1:
            pid = getattr(worker, "pid", "?")
            base = f"{role} {pid}" if role else f"pid {pid}"
        labels[id(worker)] = base
    return labels


def _ordered_lanes(spans: list[PlacedSpan]) -> list[str]:
    """Lane identities grouped by role, so a pipeline's workers sit together.

    Within a role, lanes are ordered by first activity: a reader scanning down the chart sees
    the pipeline in roughly the order work flows through it.
    """
    first_seen: dict[str, tuple[str, int]] = {}
    for span in spans:
        existing = first_seen.get(span.lane)
        if existing is None or span.t0_ns < existing[1]:
            first_seen[span.lane] = (span.role, span.t0_ns)
    return sorted(first_seen, key=lambda lane: (first_seen[lane][0], first_seen[lane][1]))


def clock_step_note(clock_steps: dict[str, int]) -> str:
    """What a rejected anchor means for the reader, or ``""`` when every clock behaved.

    Returned as a sentence rather than a flag because this qualifies the axis itself: every
    position, gap and cross-lane arrow on the page is measured along it. It belongs beside
    the headline figures, not only in the trailing caveats — the page's own history is that a
    disclosure eighty lines below a confident conclusion does not reach the person acting on
    it.
    """
    if not clock_steps:
        return ""
    workers = ", ".join(sorted(clock_steps))
    return (
        f"The wall clock stepped mid-run on: {workers}. Those readings are ignored and the "
        "spans are placed by monotonic offset from each worker's first anchor, so durations "
        "are unaffected — but absolute placement after the step, and so any gap measured "
        "between one of these lanes and another worker, is only as good as that origin."
    )


def alignment_accuracy_note(hosts: set[str], clock_stepped: bool = False) -> str:
    """How far the shared axis can be trusted, stated rather than assumed.

    Within one host the mapping is exact: every process reads the same two clocks. Across
    hosts it is only as good as the clocks agree, which on a cluster means NTP — typically
    under a millisecond, occasionally worse. Sub-millisecond arrows between two hosts are
    therefore not evidence of anything, and the page says so rather than letting a reader
    conclude otherwise.

    ``clock_stepped`` withdraws the one-host claim of exactness. Processes on one host do read
    the same clocks, and that guarantees nothing once one of those clocks moved mid-run —
    printing "exact" beside :func:`clock_step_note` would contradict it on the same page.
    """
    if len(hosts) <= 1:
        if clock_stepped:
            return (
                "All workers ran on one host and so read the same clocks, but one of those "
                "clocks stepped during the run — so the axis is exact only within a single "
                "worker's own spans, not between them."
            )
        return (
            "All workers ran on one host, so the shared time axis is exact: every process "
            "read the same clocks."
        )
    return (
        f"Workers ran on {len(hosts)} hosts. Times are aligned through each host's wall "
        "clock, so cross-host placement is only as accurate as clock synchronisation "
        "(NTP is typically well under a millisecond). Treat sub-millisecond gaps between "
        "hosts as noise."
    )
