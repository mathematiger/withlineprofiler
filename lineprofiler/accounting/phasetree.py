"""Aggregate statistics for named regions of a program, arranged as a tree.

A phase is a named region opened with :meth:`Profiler.phase`. Phases nest, so every phase
is identified by its full path from the root — ``("self_play", "mcts", "env_step")``. All
statistics are accumulated in place; nothing per-call is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lineprofiler.accounting.histogram import DurationHistogram

PhasePath = tuple[str, ...]
"""Identity of a node in the phase tree: the names of its ancestors, then its own."""


@dataclass(slots=True)
class PhaseStats:
    """Accumulated statistics for one node in the phase tree.

    ``self_ns`` is the time spent in this phase but not in any child, and ``wait_ns`` is the
    time the thread was not running on a CPU — blocked on a queue, a lock, the GIL or a
    syscall. In a queue-driven multi-process pipeline ``wait_ns`` is usually the number that
    explains the run.

    Test specifically:
        - ``merge`` is associative and commutative over three or more instances
        - ``merge`` with an empty instance is the identity
        - ``self_ns`` is never negative, including under recursive phase entry
        - ``wait_ns`` is never negative even when the CPU clock's granularity makes
          ``cpu_ns`` exceed ``wall_ns`` for very short phases
    """

    calls: int = 0
    wall_ns: int = 0
    cpu_ns: int = 0
    child_wall_ns: int = 0
    hist: DurationHistogram = field(default_factory=DurationHistogram)
    counters: dict[str, int] = field(default_factory=dict)
    sample_stride: int = 0
    """``0`` when every entry was measured; ``n`` when one entry in ``n`` was, and the totals
    above are scaled estimates rather than measurements.

    Kept as a field rather than inferred, because an estimate that cannot be told apart from a
    measurement is precisely the failure this layer exists to avoid. Merging takes the largest
    stride: one sampled contributor makes the merged node an estimate too."""

    @property
    def self_ns(self) -> int:
        """Wall time in this phase excluding time in its children."""
        return max(0, self.wall_ns - self.child_wall_ns)

    @property
    def wait_ns(self) -> int:
        """Wall time during which the thread was not executing on a CPU.

        **Pairs with ``wall_ns``, never with ``self_ns``.** Wait spans the whole phase,
        including the time spent inside children, whereas ``self_ns`` excludes it — so
        ``wait_ns / self_ns`` exceeds 100% for any parent that waits inside a child, which is
        the ordinary case for a phase wrapping a blocking call. The share the report prints
        is ``wait_ns / wall_ns``.
        """
        return max(0, self.wall_ns - self.cpu_ns)

    def record(self, wall_ns: int, cpu_ns: int) -> None:
        """Record one completed entry into this phase."""
        self.calls += 1
        self.wall_ns += wall_ns
        self.cpu_ns += cpu_ns
        self.hist.observe(wall_ns)

    def add_count(self, name: str, amount: int) -> None:
        """Attribute ``amount`` work units of kind ``name`` to this phase."""
        self.counters[name] = self.counters.get(name, 0) + amount

    def copy(self) -> PhaseStats:
        """Return a detached copy, so merging never mutates the source it was built from."""
        clone = PhaseStats()
        clone.merge(self)
        return clone

    def merge(self, other: PhaseStats) -> None:
        """Add ``other``'s totals into this instance, in place.

        ``other`` may be a node another thread is still writing to — snapshots are taken from
        a background thread while the owning threads keep recording. Iterating its counters
        directly raised ``RuntimeError: dictionary changed size during iteration`` the moment
        the owner introduced a new counter name mid-merge, which killed the flush thread.
        ``list()`` takes the items in one bytecode, so the snapshot sees a coherent set and
        anything added after it is simply picked up by the next flush.
        """
        self.calls += other.calls
        self.wall_ns += other.wall_ns
        self.cpu_ns += other.cpu_ns
        self.child_wall_ns += other.child_wall_ns
        self.hist.merge(other.hist)
        # Any sampled contributor taints the total: presenting a partly-estimated sum as
        # measured is the wrong-number failure, so the coarsest stride wins.
        self.sample_stride = max(self.sample_stride, other.sample_stride)
        for name, amount in list(other.counters.items()):
            self.add_count(name, amount)

    def difference(self, baseline: PhaseStats) -> PhaseStats:
        """Return the work recorded since ``baseline`` was taken.

        Every field is clamped at zero. A cumulative counter cannot legitimately go backwards,
        so a negative difference means the baseline came from a merge this one did not include
        — reporting it as negative work would be worse than reporting none.
        """
        result = PhaseStats(
            calls=max(0, self.calls - baseline.calls),
            wall_ns=max(0, self.wall_ns - baseline.wall_ns),
            cpu_ns=max(0, self.cpu_ns - baseline.cpu_ns),
            child_wall_ns=max(0, self.child_wall_ns - baseline.child_wall_ns),
            sample_stride=self.sample_stride,
        )
        result.hist = self.hist.difference(baseline.hist)
        for name, amount in list(self.counters.items()):
            delta = amount - baseline.counters.get(name, 0)
            if delta > 0:
                result.counters[name] = delta
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this node."""
        return {
            "calls": self.calls,
            "wall_ns": self.wall_ns,
            "cpu_ns": self.cpu_ns,
            "child_wall_ns": self.child_wall_ns,
            "hist": self.hist.to_sparse(),
            "counters": self.counters,
            "sample_stride": self.sample_stride,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseStats:
        """Rebuild a node from :meth:`to_dict` output.

        ``sample_stride`` defaults to ``0`` so a worker file written before sampling existed
        reads back as fully measured, which it was.
        """
        return cls(
            calls=data["calls"],
            wall_ns=data["wall_ns"],
            cpu_ns=data["cpu_ns"],
            child_wall_ns=data["child_wall_ns"],
            hist=DurationHistogram.from_sparse(data["hist"]),
            counters=dict(data["counters"]),
            sample_stride=data.get("sample_stride", 0),
        )


PhaseTree = dict[PhasePath, PhaseStats]
"""The whole accumulated state of one worker: every phase path it has ever entered."""


def merge_trees(target: PhaseTree, source: PhaseTree) -> None:
    """Merge ``source`` into ``target`` in place, summing shared paths.

    Nodes are copied on insert. Sharing them would alias ``source``'s statistics into
    ``target``, so a later merge into the same path would silently inflate the source's own
    totals — which is how per-worker load and the imbalance ratio get corrupted.

    Test specifically:
        - merging is associative and commutative across three or more trees
        - a path present in only one tree survives with its totals intact
        - ``source`` is unchanged by the merge, including on paths unique to it
    """
    for path, stats in source.items():
        existing = target.get(path)
        if existing is None:
            target[path] = stats.copy()
        else:
            existing.merge(stats)


def tree_to_dict(tree: PhaseTree) -> dict[str, Any]:
    """Serialise a phase tree, joining each path with ``/`` for a readable key."""
    return {"/".join(path): stats.to_dict() for path, stats in tree.items()}


def tree_from_dict(data: dict[str, Any]) -> PhaseTree:
    """Rebuild a phase tree from :func:`tree_to_dict` output.

    The root phase serialises to the empty string, which splits back to ``()``.
    """
    return {
        (tuple(key.split("/")) if key else ()): PhaseStats.from_dict(value)
        for key, value in data.items()
    }
