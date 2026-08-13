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

    @property
    def self_ns(self) -> int:
        """Wall time in this phase excluding time in its children."""
        return max(0, self.wall_ns - self.child_wall_ns)

    @property
    def wait_ns(self) -> int:
        """Wall time during which the thread was not executing on a CPU."""
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
        """Add ``other``'s totals into this instance, in place."""
        self.calls += other.calls
        self.wall_ns += other.wall_ns
        self.cpu_ns += other.cpu_ns
        self.child_wall_ns += other.child_wall_ns
        self.hist.merge(other.hist)
        for name, amount in other.counters.items():
            self.add_count(name, amount)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this node."""
        return {
            "calls": self.calls,
            "wall_ns": self.wall_ns,
            "cpu_ns": self.cpu_ns,
            "child_wall_ns": self.child_wall_ns,
            "hist": self.hist.to_sparse(),
            "counters": self.counters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseStats:
        """Rebuild a node from :meth:`to_dict` output."""
        return cls(
            calls=data["calls"],
            wall_ns=data["wall_ns"],
            cpu_ns=data["cpu_ns"],
            child_wall_ns=data["child_wall_ns"],
            hist=DurationHistogram.from_sparse(data["hist"]),
            counters=dict(data["counters"]),
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
