"""Compare two runs: what got faster, what got slower, and what changed shape.

Comparing raw wall time across runs of different lengths is meaningless, so every phase is
compared on its *per-entry* cost, and counters on their rate. A phase present in only one
run is reported as such rather than silently dropped — an added or removed phase is usually
the most interesting difference there is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lineprofiler.accounting.phase import PhaseStats
from lineprofiler.accounting.report import format_ns
from lineprofiler.accounting.snapshot import MergedRun, merge_run


@dataclass(slots=True)
class PhaseDelta:
    """One phase's change between two runs, on a per-entry basis."""

    phase: str
    calls_a: int
    calls_b: int
    per_call_a: float
    per_call_b: float

    @property
    def only_in(self) -> str | None:
        """``"A"``, ``"B"``, or ``None`` when the phase exists in both runs."""
        if self.calls_a == 0:
            return "B"
        if self.calls_b == 0:
            return "A"
        return None

    @property
    def ratio(self) -> float:
        """``per_call_b / per_call_a``; 1.0 is unchanged, 2.0 is twice as slow."""
        if not self.per_call_a or not self.per_call_b:
            return 0.0
        return self.per_call_b / self.per_call_a

    @property
    def percent(self) -> float:
        return (self.ratio - 1.0) * 100.0 if self.ratio else 0.0


def compare(run_a: MergedRun, run_b: MergedRun) -> list[PhaseDelta]:
    """Return per-phase deltas, largest regression first.

    Test specifically:
        - two runs with a known 2x ratio report +100%
        - a phase present in only one run is reported with ``only_in`` set
        - runs with different entry counts still compare correctly, since the comparison is
          per entry rather than on totals
    """
    tree_a, tree_b = run_a.tree, run_b.tree
    deltas = []
    for path in sorted({*tree_a, *tree_b}):
        if not path:
            continue
        stats_a, stats_b = tree_a.get(path), tree_b.get(path)
        deltas.append(
            PhaseDelta(
                phase="/".join(path),
                calls_a=stats_a.calls if stats_a else 0,
                calls_b=stats_b.calls if stats_b else 0,
                per_call_a=_per_call(stats_a),
                per_call_b=_per_call(stats_b),
            ),
        )
    return sorted(deltas, key=lambda delta: -delta.ratio)


def render_comparison(run_a: MergedRun, run_b: MergedRun, label_a: str, label_b: str) -> str:
    """Render the delta table as text."""
    deltas = compare(run_a, run_b)
    width = 62
    lines = [
        f"A = {label_a}",
        f"B = {label_b}",
        "",
        f"{'phase':<28}{'A /entry':>12}{'B /entry':>12}{'change':>10}",
        "─" * width,
    ]
    for delta in deltas:
        lines.append(_delta_row(delta))
    return "\n".join(lines)


def comparison_as_dict(run_a: MergedRun, run_b: MergedRun) -> dict[str, Any]:
    """Return the same comparison as JSON-serialisable data."""
    return {
        "phases": [
            {
                "phase": delta.phase,
                "calls_a": delta.calls_a,
                "calls_b": delta.calls_b,
                "per_call_ns_a": delta.per_call_a,
                "per_call_ns_b": delta.per_call_b,
                "ratio": delta.ratio,
                "percent": delta.percent,
                "only_in": delta.only_in,
            }
            for delta in compare(run_a, run_b)
        ],
    }


def compare_dirs(dir_a: str, dir_b: str) -> str:
    """Merge both run directories and render their comparison."""
    return render_comparison(merge_run(dir_a), merge_run(dir_b), dir_a, dir_b)


def _delta_row(delta: PhaseDelta) -> str:
    name = delta.phase[-27:]
    if delta.only_in == "A":
        return f"{name:<28}{format_ns(delta.per_call_a):>12}{'—':>12}{'only in A':>10}"
    if delta.only_in == "B":
        return f"{name:<28}{'—':>12}{format_ns(delta.per_call_b):>12}{'only in B':>10}"
    return (
        f"{name:<28}{format_ns(delta.per_call_a):>12}"
        f"{format_ns(delta.per_call_b):>12}{delta.percent:>+9.1f}%"
    )


def _per_call(stats: PhaseStats | None) -> float:
    if stats is None or stats.calls == 0:
        return 0.0
    return stats.wall_ns / stats.calls
