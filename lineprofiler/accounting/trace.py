"""Bounded event recording: the spans and links behind the trace timeline.

The phase tree answers "where did the time go" as a set of totals. It cannot answer "why was
this worker idle at 14:32", because a total has no position on a clock. That question needs
per-entry timestamps, which the tree deliberately does not keep — constant memory per phase
is what makes it affordable to leave on for twelve hours.

So this stores them separately, and bounds the cost by construction rather than by hoping the
run is short: a fixed-capacity ring of fixed-width integers. A twelve-hour run with tracing
on holds exactly ``capacity`` spans, the most recent ones, and says how many it dropped.
Nothing here allocates per span once the buffer is built.

Three things are deliberate and load-bearing:

- **Phase paths are interned to an integer.** Storing the tuple per span would dominate both
  memory and the flush, and the same few hundred paths repeat millions of times.
- **A wrapped buffer reports its drop count.** A truncated trace that renders as a complete
  one is the wrong-numbers failure this package exists to avoid, so ``dropped`` is carried
  all the way to the page.
- **``cpu_ns`` may be ``UNMEASURED``, never a plausible zero.** Auto-derived spans cannot
  afford a ``thread_time_ns()`` per function call, and a span whose CPU time is unknown must
  not be drawn as a span that spent none.
"""

from __future__ import annotations

import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from lineprofiler.accounting.phasetree import PhasePath

UNMEASURED = -1
"""``cpu_ns`` sentinel: this span's CPU time was not measured, so its wait is unknown.

Distinct from ``0``, which is a real measurement meaning "ran, but never on a CPU" — a fully
blocked span. Conflating them would draw an unmeasured span as 100% waiting, which is the
most misleading thing this view could say."""

FLAG_AUTO = 1
"""The span came from function-call tracing rather than a phase the user named."""

FLAG_SAMPLED = 2
"""The span's phase is entered under ``sample=``, so it represents one entry in ``n``."""

FLAG_ASYNC_UNSYNCED = 4
"""The phase declared ``async_work=True``: work it submitted was still in flight at exit.

The span's wall time is therefore submission time, not the cost of the work — and the work
itself lands on whichever later span happens to synchronise. Recorded so a renderer can say
so, because a wall time that looks like a measurement but answers a different question is the
failure this layer exists to avoid."""

DEFAULT_CAPACITY = 200_000
"""Spans retained per worker. At 48 bytes a span this is ~10 MB, which is affordable next to
a training process and deep enough to cover many thousands of iterations."""

_LINK_CAPACITY_DIVISOR = 20
"""Links are rare next to phases — a handful per iteration against hundreds of spans — so the
link ring is sized as a fraction of the span ring rather than given its own knob."""


@dataclass(slots=True)
class Span:
    """One completed phase entry, positioned on the worker's ``perf_counter`` clock.

    ``t0``/``t1`` are raw ``perf_counter_ns()`` readings, meaningful only within the process
    that took them; :mod:`lineprofiler.accounting.tracealign` maps them onto a common epoch
    using the worker's :class:`ClockAnchor` pairs.
    """

    phase_id: int
    thread_id: int
    t0_ns: int
    t1_ns: int
    cpu_ns: int
    flags: int

    @property
    def duration_ns(self) -> int:
        """Wall time this span covered."""
        return max(0, self.t1_ns - self.t0_ns)

    @property
    def cpu_measured(self) -> bool:
        """Whether ``cpu_ns`` is a measurement rather than the :data:`UNMEASURED` sentinel."""
        return self.cpu_ns != UNMEASURED

    @property
    def wait_ns(self) -> int:
        """Wall time not spent on a CPU, or ``0`` when CPU time was not measured.

        Callers must test :attr:`cpu_measured` before presenting this; an unmeasured span
        returns ``0`` so arithmetic is safe, not because it did not wait.
        """
        if not self.cpu_measured:
            return 0
        return max(0, self.duration_ns - self.cpu_ns)


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """A simultaneous reading of the monotonic and wall clocks.

    ``perf_counter_ns`` has an arbitrary per-process origin, so two workers' spans cannot be
    placed on one axis without something that ties each process's origin to a shared
    reference. One pair does that; several taken over the run also expose drift between the
    two clocks, which is what stops a long run's lanes from sliding apart.
    """

    perf_ns: int
    real_ns: int

    @classmethod
    def take(cls) -> ClockAnchor:
        """Read both clocks as close together as the interpreter allows."""
        return cls(perf_ns=time.perf_counter_ns(), real_ns=time.time_ns())

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable view."""
        return {"perf_ns": self.perf_ns, "real_ns": self.real_ns}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClockAnchor:
        """Rebuild from :meth:`to_dict` output."""
        return cls(perf_ns=int(data["perf_ns"]), real_ns=int(data["real_ns"]))


@dataclass(slots=True)
class Link:
    """One end of a cross-process dependency, named by the user at a queue boundary.

    A ``signal`` says "the thing identified by ``key`` on ``channel`` is now available"; a
    ``wait`` says "I needed it here". Matching the two across workers is what turns a lane
    full of idle time into an arrow pointing at whoever it was idle *for*.
    """

    channel: str
    key: str
    kind: str
    t_ns: int
    thread_id: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "channel": self.channel,
            "key": self.key,
            "kind": self.kind,
            "t_ns": self.t_ns,
            "thread_id": self.thread_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Link:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            channel=str(data["channel"]),
            key=str(data["key"]),
            kind=str(data["kind"]),
            t_ns=int(data["t_ns"]),
            thread_id=int(data["thread_id"]),
        )


class TraceBuffer:
    """A fixed-capacity ring of spans, plus a smaller one of links.

    Spans live in six parallel ``array("q")`` buffers rather than a list of objects: the hot
    path then does six indexed stores into preallocated memory and one integer increment,
    with no allocation and no attribute lookups on a per-span object. At roughly a
    microsecond per phase, allocating a dataclass per exit would be a visible share of it.

    Not thread-safe by design, mirroring the phase tree: each thread records into its own
    buffer and they are merged at flush, so the hot path takes no locks.

    Test specifically:
        - a buffer filled past capacity keeps the *newest* spans and reports the rest dropped
        - ``drain`` returns spans in chronological order across the wrap point
        - ``intern`` returns a stable id for a repeated path and never grows unbounded
        - draining twice does not repeat spans already taken
    """

    __slots__ = (
        "_cpu", "_flags", "_link_capacity", "_links", "_paths", "_phase", "_t0", "_t1",
        "_thread", "capacity", "dropped", "dropped_links", "path_ids", "wrapped", "write",
    )

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError(f"trace capacity must be positive, got {capacity!r}")
        self.capacity = capacity
        self._phase = array("q", [0]) * capacity
        self._thread = array("q", [0]) * capacity
        self._t0 = array("q", [0]) * capacity
        self._t1 = array("q", [0]) * capacity
        self._cpu = array("q", [0]) * capacity
        self._flags = array("q", [0]) * capacity
        self.write = 0
        self.wrapped = False
        self.dropped = 0
        self.path_ids: dict[PhasePath, int] = {}
        self._paths: list[PhasePath] = []
        self._link_capacity = max(16, capacity // _LINK_CAPACITY_DIVISOR)
        self._links: deque[Link] = deque(maxlen=self._link_capacity)
        self.dropped_links = 0

    def intern(self, path: PhasePath) -> int:
        """Return a stable integer id for ``path``, creating one on first sight.

        Called only when a phase path is first traced, never per span: the caller holds the
        id alongside the phase's stats. Bounded by ``MAX_PHASES`` upstream, which is what
        keeps this table from growing with a generated name.
        """
        existing = self.path_ids.get(path)
        if existing is not None:
            return existing
        assigned = len(self._paths)
        self.path_ids[path] = assigned
        self._paths.append(path)
        return assigned

    def record(
        self,
        phase_id: int,
        thread_id: int,
        t0_ns: int,
        t1_ns: int,
        cpu_ns: int,
        flags: int = 0,
    ) -> None:
        """Append one span, overwriting the oldest when full.

        This is the hot path. It must stay allocation-free and branch-light: one compare,
        six stores, one increment.

        ``dropped`` counts spans this write *overwrote*, which is why it is incremented
        before the store and only once the ring has wrapped. Counting on the write that
        merely fills the last slot would report one loss that never happened.
        """
        index = self.write
        if self.wrapped:
            self.dropped += 1
        self._phase[index] = phase_id
        self._thread[index] = thread_id
        self._t0[index] = t0_ns
        self._t1[index] = t1_ns
        self._cpu[index] = cpu_ns
        self._flags[index] = flags
        index += 1
        if index >= self.capacity:
            index = 0
            self.wrapped = True
        self.write = index

    def record_link(self, channel: str, key: str, kind: str, thread_id: int) -> None:
        """Append one dependency endpoint, stamped with the current monotonic clock.

        Links are far rarer than spans, so this holds objects rather than parallel arrays;
        the clarity is worth more than the allocation at this frequency.
        """
        # A deque with maxlen, not a list: the old `pop(0)` shifted every surviving link on
        # each overflow, which was tolerable while links were a handful per iteration and is
        # not now that a request lifecycle records several marks per request. Eviction is O(1)
        # and keeps the newest, exactly as the span ring does.
        if len(self._links) >= self._link_capacity:
            self.dropped_links += 1
        self._links.append(
            Link(
                channel=channel,
                key=key,
                kind=kind,
                t_ns=time.perf_counter_ns(),
                thread_id=thread_id,
            ),
        )

    def paths(self) -> list[PhasePath]:
        """The interning table, indexed by the ids stored in spans."""
        return list(self._paths)

    def drain(self) -> tuple[list[Span], list[Link]]:
        """Remove and return everything recorded so far, oldest first.

        Draining empties the ring, so a periodic flush writes each span exactly once and the
        capacity bounds what is held in memory between flushes rather than for the whole run.
        The interning table is *kept*: ids already written to disk must keep their meaning.
        """
        spans = [
            Span(
                phase_id=self._phase[index],
                thread_id=self._thread[index],
                t0_ns=self._t0[index],
                t1_ns=self._t1[index],
                cpu_ns=self._cpu[index],
                flags=self._flags[index],
            )
            for index in self._live_indices()
        ]
        # list(), and a fresh deque rather than clear(): callers serialise the result and
        # must not hold a view onto the buffer that keeps filling behind them.
        links = list(self._links)
        self._links = deque(maxlen=self._link_capacity)
        self.write = 0
        self.wrapped = False
        return spans, links

    def _live_indices(self) -> range | list[int]:
        """Indices holding recorded spans, in chronological order.

        After a wrap the oldest live span sits at the write cursor, so the ring reads as two
        runs: cursor to end, then start to cursor.
        """
        if not self.wrapped:
            return range(self.write)
        return [*range(self.write, self.capacity), *range(self.write)]

    def is_empty(self) -> bool:
        """Whether anything has been recorded since the last drain."""
        return self.write == 0 and not self.wrapped and not self._links

    def clear(self) -> None:
        """Drop everything, including the interning table.

        Used after a fork: the child inherits the parent's buffer contents, which describe
        work the child never did, and inherited path ids would collide with the ones the
        child assigns.
        """
        self.write = 0
        self.wrapped = False
        self.dropped = 0
        self.dropped_links = 0
        self.path_ids = {}
        self._paths = []
        self._links = deque(maxlen=self._link_capacity)
        self.dropped_links = 0


@dataclass(slots=True)
class WorkerTrace:
    """Everything one worker recorded, as read back from its sidecar file."""

    spans: list[Span] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    anchors: list[ClockAnchor] = field(default_factory=list)
    paths: list[PhasePath] = field(default_factory=list)
    dropped: int = 0
    dropped_links: int = 0

    def path_of(self, phase_id: int) -> PhasePath:
        """Return the phase path a span's id refers to, or a placeholder when unknown.

        A span whose id is missing from the table means a torn or partial sidecar file. It is
        shown as ``(unknown)`` rather than discarded, because dropping it would quietly
        shorten the very lane the reader is trying to explain.
        """
        if 0 <= phase_id < len(self.paths):
            return self.paths[phase_id]
        return ("(unknown)",)
