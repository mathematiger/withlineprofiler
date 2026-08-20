"""What the timeline *means*, ranked: the page's conclusion rather than its evidence.

Every other derivation in this package hands the reader material and lets them synthesise:
the lane table says a worker was open 100% and on a CPU 45%, the critical path lists a chain,
the phase tree gives totals. All of it is true and none of it says "the learner spent half the
run waiting for the slowest actor". Someone who already knows the run can read that off; the
person the report was mailed to cannot, and they are who the page is for.

So this module names bottlenecks in one sentence each and orders them by how much of the run
they cost. It is pure derivation over an :class:`AlignedTrace` — no rendering, no I/O — so the
text report, the HTML page and a CI gate can all reach the same conclusions from the same
numbers, and a finding can be unit-tested without parsing HTML.

Two rules hold everywhere here:

- **A finding must be falsifiable.** Each carries the figures it was derived from, so a reader
  who distrusts the sentence can check the arithmetic rather than take it on faith.
- **Unmeasured is never reported as zero.** A lane whose CPU time was never sampled has an
  unknown wait, not a wait of nothing, and it is excluded from the ranking rather than shown
  at the bottom of it. The ``-1.0`` convention comes from :mod:`tracealign`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lineprofiler.accounting.trace import Origin
from lineprofiler.accounting.tracealign import (
    AlignedTrace,
    PlacedSpan,
    concurrent_activity,
    lane_busy_share,
    lane_working_share,
)

_MAX_FINDINGS = 6
"""Findings shown. Past a handful this stops being a conclusion and becomes another list."""

_IDLE_LANE_PCT = 25.0
"""Idle share of the whole trace below which a lane is not worth calling out.

A worker that starts late or finishes early is idle at the edges by construction, and a page
that flags every such lane trains the reader to skip the section that is supposed to matter.
"""

_BLOCKED_PHASE_PCT = 20.0
"""Share of a phase's own wall time spent blocked before the phase is worth naming."""

_MIN_SHARE_OF_RUN = 5.0
"""Share of the whole trace a phase must cover before its blocking is worth a finding.

A phase that is 99% blocked but runs for a millisecond in a five-minute run is a curiosity,
not a bottleneck, and ranking by blocked-share alone would put it above the real answer.
"""

_CONCURRENT_ROLE_PCT = 50.0
"""Share of a wait another role must be busy for before the wait reads as a queue.

Below this the waiter was blocked while the rest of the run was largely idle, which is a
stall — nobody was working on what it needed — and the two have nothing in common but the
symptom. The distinction is the single most useful thing this module says.
"""


@dataclass(frozen=True, slots=True)
class Finding:
    """One ranked statement about where the run lost time.

    ``headline`` is the sentence; ``detail`` is why it is true, in the same breath, so the
    reader never has to hold a number from one section against a number from another.
    ``cost_pct`` is share of the whole traced span, and is what the ranking sorts on — every
    finding is measured against the same denominator so they can be compared at all.

    ``anchor`` names what to look at on the timeline: a lane id or a phase path, which the
    page turns into a control that jumps there. A finding the reader cannot act on is a
    complaint.
    """

    kind: str
    headline: str
    detail: str
    cost_pct: float
    anchor: str = ""
    lanes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_actionable(self) -> bool:
        """Whether the page can point the reader at something for this finding."""
        return bool(self.anchor)


def rank_findings(trace: AlignedTrace) -> list[Finding]:
    """Everything worth saying about this trace, most expensive first.

    Returns an empty list for a run with nothing wrong, which is a legitimate answer and is
    rendered as such: "no single phase dominates" is information, and inventing a finding to
    fill the section would make every page look like it had a problem.
    """
    if not trace.spans or trace.duration_ns <= 0:
        return []

    findings = [
        *_blocked_phase_findings(trace),
        *_idle_lane_findings(trace),
        *_serial_findings(trace),
    ]
    findings.sort(key=lambda finding: -finding.cost_pct)
    return findings[:_MAX_FINDINGS]


def _blocked_phase_findings(trace: AlignedTrace) -> list[Finding]:
    """Phases that spent their time blocked, and who was working while they did.

    This is the finding the page exists for. The wait is attributed to the phase that was
    open, and the queue/stall split turns it into either "the work existed and this waited its
    turn" or "nothing was being produced at all".

    Only the *innermost* blocked phase of a chain is reported. A parent that does nothing but
    call a blocking child inherits all of its wait, so reporting both says one problem twice
    and pushes the real answer down the ranking — the reader is told ``iteration`` is 55%
    blocked and ``iteration/queue_get`` is 54% blocked, which is one fact wearing two hats.
    """
    by_path = _measured_spans_by_path(trace)
    candidates: dict[str, Finding] = {}
    for path, spans in by_path.items():
        wall = sum(span.duration_ns for span in spans)
        wait = sum(span.wait_ns for span in spans)
        if wall <= 0:
            continue
        blocked_pct = 100.0 * wait / wall
        share_of_run = 100.0 * wall / trace.duration_ns
        if blocked_pct < _BLOCKED_PHASE_PCT or share_of_run < _MIN_SHARE_OF_RUN:
            continue
        candidates[path] = _blocked_phase_finding(trace, path, spans, wait, blocked_pct)
    return [
        finding
        for path, finding in candidates.items()
        if not _has_blocked_descendant(path, candidates)
    ]


def _has_blocked_descendant(path: str, candidates: dict[str, Finding]) -> bool:
    """Whether a deeper phase already accounts for this one's wait.

    Containment by path prefix, which is exact for named phases: a phase path *is* its call
    stack, so ``iteration/queue_get`` being present means ``iteration``'s wait was spent
    inside it and reporting the parent adds nothing.
    """
    prefix = f"{path}/"
    return any(other.startswith(prefix) for other in candidates)


def _blocked_phase_finding(
    trace: AlignedTrace,
    path: str,
    spans: list[PlacedSpan],
    wait_ns: int,
    blocked_pct: float,
) -> Finding:
    """One blocked phase, classified as a queue or a stall by what else was running."""
    cost_pct = 100.0 * wait_ns / trace.duration_ns
    lanes = tuple(sorted({span.lane for span in spans}))
    blocked = [span for span in spans if span.wait_ns > 0]
    detail = _explain_wait(trace, spans, blocked)

    return Finding(
        kind="blocked-phase",
        headline=(
            f"{path} spent {blocked_pct:.0f}% of its time blocked, "
            f"costing {cost_pct:.0f}% of the run"
        ),
        detail=detail,
        cost_pct=cost_pct,
        anchor=path,
        lanes=lanes,
    )


def _explain_wait(
    trace: AlignedTrace,
    spans: list[PlacedSpan],
    blocked: list[PlacedSpan],
) -> str:
    """Say whether a wait was a queue or a stall, preferring recorded evidence to inference.

    Two sources, in order of authority:

    1. **A matched arrow.** ``signal``/``wait_on`` records who released this wait and when.
       That is not an inference — the producer is named — so it settles the question outright.
    2. **Concurrent activity.** Absent arrows, the fallback asks how busy everyone else was
       while this blocked. It is weaker on purpose: a role working in short bursts across a
       long wait scores low against the wait's union even though it was producing throughout,
       so a low figure here means "probably a stall", never "certainly".

    Getting the order wrong is what made the learner's ``queue_get`` — released by an actor's
    ``signal`` on every single iteration — report as a stall because the actors' bursts only
    covered a quarter of its wait.
    """
    releaser, delay_share = _releasing_role(trace, spans)
    if releaser:
        return (
            f"released by {releaser} on a recorded signal/wait_on pair, so this is a queue, "
            f"not a hang: the work existed and this phase waited its turn. "
            f"{delay_share:.0f}% of the wait was after the producer had already signalled."
        )

    if not blocked:
        return "No span on this phase recorded a wait, so there is nothing to attribute."

    activity = concurrent_activity(trace, blocked)
    if not activity:
        return (
            "No other lane had a phase open during that wait, so nothing was being produced "
            "while this blocked — a stall rather than a queue."
        )

    role, share = max(activity.items(), key=lambda item: item[1])
    if share >= _CONCURRENT_ROLE_PCT:
        return (
            f"{role} had a phase open for {share:.0f}% of that wait, so this is a queue — the "
            f"work existed and this phase was waiting its turn, not hung."
        )
    return (
        f"The busiest other role ({role}) had a phase open for only {share:.0f}% of that "
        f"wait, so this looks closer to a stall than a queue. No signal/wait_on pair was "
        f"recorded for it, so that is inferred from concurrency rather than measured."
    )


def _releasing_role(trace: AlignedTrace, spans: list[PlacedSpan]) -> tuple[str, float]:
    """The role whose ``signal`` released this phase's waits, and the share spent post-signal.

    A wait that ends when a producer signals is a queue by definition. The second figure
    separates two very different queues: time before the signal is the producer being slow,
    time after it is the waiter being slow to wake — scheduling pressure rather than
    throughput.
    """
    lanes = {span.lane for span in spans}
    workers = {lane.split("#", 1)[0] for lane in lanes}
    matched = [arrow for arrow in trace.arrows if arrow.dst_worker in workers]
    if not matched:
        return "", 0.0

    by_role: dict[str, int] = {}
    for arrow in matched:
        role = trace.roles.get(f"{arrow.src_worker}#0", "") or arrow.src_worker
        by_role[role] = by_role.get(role, 0) + 1
    role = max(by_role.items(), key=lambda item: item[1])[0]

    total_delay = sum(arrow.delay_ns for arrow in matched)
    waited = sum(span.wait_ns for span in spans)
    share = 100.0 * total_delay / waited if waited > 0 else 0.0
    return role, min(100.0, share)


def _idle_lane_findings(trace: AlignedTrace) -> list[Finding]:
    """Lanes that had no phase open for much of the run — capacity nobody used.

    Distinct from a blocked phase: that lane was inside a call and waiting, this one was not
    inside anything at all. On a fixed worker pool the two have different fixes, so the page
    must not collapse them into one number.
    """
    idle_by_role: dict[str, list[tuple[str, float, float]]] = {}
    for lane in trace.lanes:
        idle = 100.0 - lane_busy_share(trace, lane)
        if idle < _IDLE_LANE_PCT:
            continue
        role = trace.roles.get(lane, "")
        idle_by_role.setdefault(role, []).append(
            (lane, idle, lane_working_share(trace, lane)),
        )

    return [_idle_role_finding(trace, role, lanes) for role, lanes in idle_by_role.items()]


def _idle_role_finding(
    trace: AlignedTrace,
    role: str,
    idle_lanes: list[tuple[str, float, float]],
) -> Finding:
    """One statement per role, not per lane.

    Sixteen actors idle for the same reason is one finding about the pipeline, not sixteen
    about the actors; listing them individually fills the section and pushes out everything
    that is not about them. The worst lane is named so there is still something to click.
    """
    worst_lane, worst_idle, _ = max(idle_lanes, key=lambda entry: entry[1])
    mean_idle = sum(idle for _, idle, _ in idle_lanes) / len(idle_lanes)
    measured = [working for _, _, working in idle_lanes if working >= 0]
    cpu = (
        f" Those lanes were on a CPU for {sum(measured) / len(measured):.0f}% of the run."
        if measured
        else " Their CPU time was never measured, so how much of the rest was work is unknown."
    )

    subject = (
        f"{len(idle_lanes)} {role or 'unlabelled'} lanes had no phase open for "
        f"{mean_idle:.0f}% of the run on average"
        if len(idle_lanes) > 1
        else f"{worst_lane} had no phase open for {worst_idle:.0f}% of the run"
        + (f" ({role})" if role else "")
    )
    return Finding(
        kind="idle-lane",
        headline=subject,
        detail=(
            "Idle time is drawn as absence on the timeline, so these lanes read as rows full "
            "of gaps." + cpu
        ),
        cost_pct=mean_idle * len(idle_lanes) / max(1, len(trace.lanes)),
        anchor=worst_lane,
        lanes=tuple(lane for lane, _, _ in idle_lanes),
    )


def _serial_findings(trace: AlignedTrace) -> list[Finding]:
    """Stretches where exactly one lane was working and every other one was empty.

    On a run with several workers this is the clearest possible statement of lost parallelism:
    the machine had *n* lanes and used one. Reported only where there is more than one lane,
    since "one lane was working" is not a finding about a single-lane run.
    """
    if len(trace.lanes) < 2:
        return []
    serial_ns = _serial_time_ns(trace)
    share = 100.0 * serial_ns / trace.duration_ns
    if share < _MIN_SHARE_OF_RUN:
        return []
    return [
        Finding(
            kind="serial",
            headline=(
                f"only one of {len(trace.lanes)} lanes was active for {share:.0f}% of the run"
            ),
            detail=(
                "For that share of the traced span every other lane had nothing open, so the "
                "extra workers bought nothing there. Look for the widest single bar with "
                "empty rows beside it."
            ),
            cost_pct=share,
        ),
    ]


def _serial_time_ns(trace: AlignedTrace) -> int:
    """Nanoseconds during which exactly one lane had a phase open.

    Computed by sweeping the span boundaries rather than by sampling: a sampled estimate of
    parallelism is wrong in exactly the bursty regions the reader cares about most.
    """
    events: list[tuple[int, int, str]] = []
    for span in trace.spans:
        events.append((span.t0_ns, 1, span.lane))
        events.append((span.t1_ns, -1, span.lane))
    events.sort(key=lambda event: event[0])

    open_by_lane: dict[str, int] = {}
    serial = 0
    previous = events[0][0] if events else 0
    for at, delta, lane in events:
        active = sum(1 for count in open_by_lane.values() if count > 0)
        if active == 1:
            serial += at - previous
        previous = at
        open_by_lane[lane] = open_by_lane.get(lane, 0) + delta
    return serial


def _measured_spans_by_path(trace: AlignedTrace) -> dict[str, list[PlacedSpan]]:
    """Spans grouped by full phase path, keeping only those whose CPU time was measured.

    Auto-derived spans are excluded rather than counted as never having waited: their wait is
    unknown, and a finding built on them would state a confident figure about data that does
    not exist.
    """
    by_path: dict[str, list[PlacedSpan]] = {}
    for span in trace.spans:
        if not span.cpu_measured:
            continue
        by_path.setdefault("/".join(span.path), []).append(span)
    return by_path


@dataclass(frozen=True, slots=True)
class PhaseTotal:
    """One row of the phase summary: a phase name's cost across every lane it ran on.

    Vampir's Function Summary, in the vocabulary this package already uses. The timeline shows
    *when*; this shows *how much*, which is the question a reader opening a profile actually
    starts with and the one a timeline answers worst.
    """

    path: str
    calls: int
    wall_ns: int
    wait_ns: int
    self_ns: int
    lanes: int
    measured: bool
    origin: Origin | None = None
    """Where this phase's code is defined, when every span agreed on one place.

    A summary row is the first thing a reader ranks by, so the heaviest row naming its own
    file is what turns "``step`` is 71% of the run" into somewhere to go. ``None`` where the
    spans came from a named phase, and also where one path spans several definitions — see
    :func:`_shared_origin`."""

    @property
    def wait_pct(self) -> float:
        """Share of this phase's wall time spent blocked, ``-1.0`` when unmeasured."""
        if not self.measured or self.wall_ns <= 0:
            return -1.0
        return 100.0 * self.wait_ns / self.wall_ns


def phase_totals(trace: AlignedTrace) -> list[PhaseTotal]:
    """Every phase path with its totals, heaviest first.

    ``self_ns`` excludes time inside nested phases on the same lane, so a wrapper phase does
    not out-rank the callee that actually spent the time. Ranked by wall rather than self,
    because the first question is "what is this run made of" — the self column is there for
    the second.
    """
    self_by_span = _self_times(trace)
    grouped: dict[str, list[PlacedSpan]] = {}
    for span in trace.spans:
        grouped.setdefault("/".join(span.path), []).append(span)

    totals = [
        _phase_total(path, spans, self_by_span) for path, spans in grouped.items()
    ]
    totals.sort(key=lambda total: -total.wall_ns)
    return totals


def _phase_total(
    path: str,
    spans: list[PlacedSpan],
    self_by_span: dict[int, int],
) -> PhaseTotal:
    """Aggregate one phase path's spans into a summary row."""
    measured = [span for span in spans if span.cpu_measured]
    return PhaseTotal(
        path=path,
        calls=len(spans),
        wall_ns=sum(span.duration_ns for span in spans),
        wait_ns=sum(span.wait_ns for span in measured),
        self_ns=sum(self_by_span.get(id(span), span.duration_ns) for span in spans),
        lanes=len({span.lane for span in spans}),
        measured=bool(measured),
        origin=_shared_origin(spans),
    )


def _shared_origin(spans: list[PlacedSpan]) -> Origin | None:
    """The one place these spans came from, or ``None`` when they disagree.

    Disagreement is real: two modules can define functions of the same qualname, and their
    spans then share a phase path. Naming either file on that row would send the reader to
    code that accounts for only part of the time, so the row says nothing instead — the
    timeline still carries each span's own location.
    """
    seen = {span.origin for span in spans if span.origin is not None}
    if len(seen) != 1:
        return None
    return seen.pop()


def _self_times(trace: AlignedTrace) -> dict[int, int]:
    """Every span's wall time minus the time its direct children covered, in one pass.

    Keyed by ``id`` because :class:`PlacedSpan` is frozen but not hashable by value — two
    genuinely distinct calls can carry identical fields, and collapsing them would silently
    merge a phase entered twice with the same timings.

    Children are found by containment on the same lane at the next depth rather than by phase
    path: an auto-derived span's path is a qualname carrying no ancestry, so the path cannot
    be trusted to describe the call structure. Sorting each lane once and walking it with a
    stack keeps this linear — the pairwise search it replaced was quadratic, which at the
    120k-span drawing cap is billions of comparisons for a column of the summary table.
    """
    child_ns: dict[int, int] = {}
    by_lane: dict[str, list[PlacedSpan]] = {}
    for span in trace.spans:
        by_lane.setdefault(span.lane, []).append(span)

    for spans in by_lane.values():
        # Start ascending, depth ascending: a container is always seen before what it contains,
        # so the stack top is the innermost span still open at this point on the lane.
        ordered = sorted(spans, key=lambda span: (span.t0_ns, span.depth))
        stack: list[PlacedSpan] = []
        for span in ordered:
            while stack and stack[-1].t1_ns <= span.t0_ns:
                stack.pop()
            if stack and stack[-1].depth == span.depth - 1:
                parent = stack[-1]
                child_ns[id(parent)] = child_ns.get(id(parent), 0) + span.duration_ns
            stack.append(span)

    return {
        id(span): max(0, span.duration_ns - child_ns.get(id(span), 0))
        for span in trace.spans
    }
