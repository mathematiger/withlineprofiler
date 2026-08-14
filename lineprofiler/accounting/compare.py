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

THIN_EVIDENCE = 30
"""Entries below which a row is marked thin. A conventional small-sample threshold, not a
computed one: the histogram gives quantiles but not the variance a real test would need."""

_TAIL_DIVERGENCE = 0.25
"""How far the mean and median ratios may drift apart before the change is called a tail
change rather than a shift."""


@dataclass(slots=True)
class PhaseDelta:
    """One phase's change between two runs, on a per-entry basis.

    Both the mean and the median are carried, because they fail differently. The mean is what
    "per entry" naturally means and it is what the totals reconcile to, but it is also the
    statistic most sensitive to the tail this layer works hard to capture: one 9-second stall
    in ten thousand 1 ms entries moves it visibly. The median comes from the merged histogram
    and ignores that stall entirely. When the two disagree, the change is in the tail, and
    the report says so rather than making you infer it.
    """

    phase: str
    calls_a: int
    calls_b: int
    per_call_a: float
    per_call_b: float
    p50_a: float = 0.0
    p50_b: float = 0.0

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
    def p50_ratio(self) -> float:
        """The same comparison on medians, which the tail cannot move."""
        if not self.p50_a or not self.p50_b:
            return 0.0
        return self.p50_b / self.p50_a

    @property
    def percent(self) -> float:
        return (self.ratio - 1.0) * 100.0 if self.ratio else 0.0

    @property
    def samples(self) -> int:
        """The smaller of the two entry counts — what the comparison actually rests on."""
        return min(self.calls_a, self.calls_b)

    @property
    def is_thin(self) -> bool:
        """Whether too few entries back this row to read the change as a finding.

        Not a significance test — the data to run one is not retained — but it separates
        "measured ten thousand times" from "measured twice", which the table used to present
        with identical authority.
        """
        return self.samples < THIN_EVIDENCE

    @property
    def tail_moved(self) -> bool:
        """Whether mean and median disagree enough that the change is in the distribution.

        A phase whose median is flat while its mean doubled did not get slower; it grew a
        tail, which is a different bug with a different fix.
        """
        if not (self.ratio and self.p50_ratio):
            return False
        return abs(self.ratio - self.p50_ratio) > _TAIL_DIVERGENCE


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
                p50_a=_median(stats_a),
                p50_b=_median(stats_b),
            ),
        )
    # Phases present in only one run sort first: an added or removed phase is usually the
    # most interesting difference there is, and their ratio of 0.0 used to bury them below
    # every improvement.
    return sorted(deltas, key=lambda delta: (delta.only_in is None, -delta.ratio))


def render_comparison(run_a: MergedRun, run_b: MergedRun, label_a: str, label_b: str) -> str:
    """Render the delta table as text."""
    deltas = compare(run_a, run_b)
    lines = [
        f"A = {label_a}",
        f"B = {label_b}",
        "",
        f"{'phase':<28}{'A /entry':>12}{'B /entry':>12}{'change':>10}{'n':>8}",
        "─" * 70,
    ]
    for delta in deltas:
        lines.append(_delta_row(delta))
    lines.extend(_comparison_notes(deltas))
    return "\n".join(lines)


def _comparison_notes(deltas: list[PhaseDelta]) -> list[str]:
    """Say what the table cannot: which rows are thin, and which moved only in the tail."""
    thin = [d for d in deltas if d.only_in is None and d.is_thin]
    tails = [d for d in deltas if d.only_in is None and d.tail_moved]
    notes: list[str] = []
    if thin:
        notes.append("")
        notes.append(f"  ? = fewer than {THIN_EVIDENCE} entries in one run; too thin to read")
        notes.append("      as a regression. These are means, not a significance test.")
    if tails:
        notes.append("")
        notes.append("  ~ = the mean moved but the median did not (or vice versa), so the")
        notes.append("      change is in the tail rather than a shift of the whole phase:")
        for delta in tails[:4]:
            notes.append(
                f"      {delta.phase[-24:]:<24}"
                f"mean {delta.ratio:.2f}x   median {delta.p50_ratio:.2f}x",
            )
    return notes


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
                "p50_ns_a": delta.p50_a,
                "p50_ns_b": delta.p50_b,
                "ratio": delta.ratio,
                "p50_ratio": delta.p50_ratio,
                "percent": delta.percent,
                "samples": delta.samples,
                "thin": delta.is_thin,
                "tail_moved": delta.tail_moved,
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
        return (
            f"{name:<28}{format_ns(delta.per_call_a):>12}{'—':>12}"
            f"{'only in A':>10}{delta.calls_a:>8}"
        )
    if delta.only_in == "B":
        return (
            f"{name:<28}{'—':>12}{format_ns(delta.per_call_b):>12}"
            f"{'only in B':>10}{delta.calls_b:>8}"
        )
    mark = "?" if delta.is_thin else ("~" if delta.tail_moved else " ")
    return (
        f"{name:<28}{format_ns(delta.per_call_a):>12}"
        f"{format_ns(delta.per_call_b):>12}{delta.percent:>+8.1f}%{mark}{delta.samples:>7}"
    )


def _per_call(stats: PhaseStats | None) -> float:
    if stats is None or stats.calls == 0:
        return 0.0
    return stats.wall_ns / stats.calls


def _median(stats: PhaseStats | None) -> float:
    """The phase's p50, straight from the merged histogram — already there, previously unused."""
    if stats is None or stats.hist.count == 0:
        return 0.0
    return float(stats.hist.quantile(0.5))
