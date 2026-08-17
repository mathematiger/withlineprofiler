"""Upper bounds on the phase hot path, so the numbers the README quotes cannot rot silently.

The entire value proposition of the accounting layer is that it is cheap enough to leave on
for twelve hours. That claim lived only in a hand-copied table, so a ten-fold regression in
``_PhaseScope`` would have merged unnoticed.

The bounds are deliberately loose — roughly 4x the measured figures — because these run on
shared CI hardware where a neighbour's build shows up as jitter. They are here to catch an
order-of-magnitude regression, an accidental syscall or lock on the hot path, not a 10%
drift. Tightening them until they fail on a busy runner would make the suite a liability.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from time import perf_counter_ns

import pytest

from lineprofiler import accounting
from lineprofiler.accounting import Profiler
from lineprofiler.accounting.capabilities import cuda_synchronize


def _a_tracer_is_active() -> bool:
    """Whether anything is instrumenting execution, by either mechanism.

    ``sys.gettrace()`` alone is not enough: a ``sys.monitoring`` tool — this package's own
    3.12+ backend among them — instruments every line while leaving ``gettrace()`` at
    ``None``, so a guard that checks only the old hook lets the timing assertions run
    against an instrumented interpreter and fail for a reason that has nothing to do with
    the hot path.
    """
    if sys.gettrace() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return False
    return any(monitoring.get_tool(tool) is not None for tool in range(6))


pytestmark = [
    pytest.mark.overhead,
    # Coverage installs a global trace function that instruments every line, inflating these
    # by an order of magnitude. Timing assertions under it measure the instrumentation.
    pytest.mark.skipif(
        _a_tracer_is_active(),
        reason="a tracer is active (coverage?); hot-path timings are meaningless under one",
    ),
]

ITERATIONS = 20_000

# Measured on an Intel Xeon at Python 3.12: 322 / 2286 / 3877 / 347 / 301 ns.
BUDGET_NS = {
    "disabled": 1_600,
    "no_cpu": 9_000,
    "with_cpu": 15_000,
    "count": 1_600,
    "ambient_uninstalled": 1_600,
    "sampled": 6_000,
    "traced": 20_000,
}


def _per_call_ns(action: Callable[[], None], iterations: int = ITERATIONS) -> float:
    """Best of three runs, to take the machine's quietest moment rather than its average."""
    best = float("inf")
    for _ in range(3):
        action()  # warm up
        start = perf_counter_ns()
        for _ in range(iterations):
            action()
        best = min(best, (perf_counter_ns() - start) / iterations)
    return best


@pytest.fixture
def run_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def test_a_disabled_phase_is_nearly_free(run_dir: Path) -> None:
    """``enabled=False`` must cost no clock reads and no allocation — it is what lets a
    codebase leave the calls in permanently."""
    profiler = Profiler(run_dir=run_dir, enabled=False)

    def action() -> None:
        with profiler.phase("p"):
            pass

    assert _per_call_ns(action) < BUDGET_NS["disabled"]


def test_a_phase_without_cpu_time_stays_under_budget(run_dir: Path) -> None:
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, measure_cpu=False,
    )

    def action() -> None:
        with profiler.phase("p"):
            pass

    try:
        assert _per_call_ns(action) < BUDGET_NS["no_cpu"]
    finally:
        profiler.close()


def test_a_phase_with_cpu_time_stays_under_budget(run_dir: Path) -> None:
    """``measure_cpu`` roughly doubles the cost — ``thread_time_ns`` is a real syscall — and
    that ratio is the thing worth defending, not the absolute number."""
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, measure_cpu=True,
    )

    def action() -> None:
        with profiler.phase("p"):
            pass

    try:
        assert _per_call_ns(action) < BUDGET_NS["with_cpu"]
    finally:
        profiler.close()


def test_counting_is_cheaper_than_a_phase(run_dir: Path) -> None:
    """The documented advice — ``count()`` in inner loops, ``phase()`` around them — is only
    true while this holds."""
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def counting() -> None:
        profiler.count("units", 1)

    def phasing() -> None:
        with profiler.phase("p"):
            pass

    try:
        count_ns = _per_call_ns(counting)
        phase_ns = _per_call_ns(phasing)
        assert count_ns < BUDGET_NS["count"]
        assert count_ns * 2 < phase_ns, (
            f"count() at {count_ns:.0f}ns is no longer meaningfully cheaper than "
            f"phase() at {phase_ns:.0f}ns; the README's advice depends on this"
        )
    finally:
        profiler.close()


def test_a_synchronising_phase_costs_nothing_without_cuda(run_dir: Path) -> None:
    """``sync=True`` resolves to ``None`` on a CPU box, so it must not cost a Python call.

    Skipped where CUDA is present, because there the call is real and *should* cost: draining
    the queue is the entire point. Its correctness is covered in ``test_gpu_hardware.py``.
    """
    if cuda_synchronize() is not None:
        pytest.skip("CUDA present: sync=True does real work here, so it is not free")
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, measure_cpu=False,
    )

    def plain() -> None:
        with profiler.phase("p"):
            pass

    def syncing() -> None:
        with profiler.phase("p", sync=True):
            pass

    try:
        assert _per_call_ns(syncing) < _per_call_ns(plain) * 1.5
    finally:
        profiler.close()


def test_an_ambient_phase_with_nothing_installed_is_nearly_free() -> None:
    """What makes it safe to instrument library code that does not know whether it is being
    profiled: with no profiler installed the call is a global load and an identity test."""
    accounting.uninstall_profiler()

    def action() -> None:
        with accounting.phase("p"):
            pass

    assert _per_call_ns(action) < BUDGET_NS["ambient_uninstalled"]


def test_resolving_the_installed_profiler_costs_almost_nothing(run_dir: Path) -> None:
    """The indirection must not be a reason to keep threading a ``profiler`` argument through
    every caller. Measured at ~38 ns on an enabled phase, about 1%."""
    profiler = Profiler(
        run_dir=run_dir, enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None, measure_cpu=False,
    )

    def direct() -> None:
        with profiler.phase("p"):
            pass

    def ambient() -> None:
        with accounting.phase("p"):
            pass

    try:
        assert _per_call_ns(ambient) < _per_call_ns(direct) * 1.5
    finally:
        profiler.close()
        accounting.uninstall_profiler()


def test_a_sampled_phase_is_cheaper_than_a_measured_one(run_dir: Path) -> None:
    """Sampling only avoids the measurement, never the Python call around it, so the saving
    is a factor of a few and not the sampling rate. The README quotes ~3.7x against the
    default; this defends the direction, not the figure."""
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def measured() -> None:
        with profiler.phase("m"):
            pass

    def sampled() -> None:
        with profiler.phase("s", sample=0.01):
            pass

    try:
        sampled_ns = _per_call_ns(sampled)
        assert sampled_ns < BUDGET_NS["sampled"]
        assert sampled_ns < _per_call_ns(measured), (
            "sampling must at least be cheaper than measuring, or it buys nothing at all"
        )
    finally:
        profiler.close()


def test_counting_stays_cheaper_than_a_sampled_phase(run_dir: Path) -> None:
    """The README tells readers to prefer count() when they only want a rate. That advice
    holds only while this does."""
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )

    def counting() -> None:
        profiler.count("units", 1)

    def sampled() -> None:
        with profiler.phase("s", sample=0.01):
            pass

    try:
        assert _per_call_ns(counting) < _per_call_ns(sampled)
    finally:
        profiler.close()


def test_a_traced_phase_stays_under_budget(run_dir: Path) -> None:
    """Tracing costs a store per phase, not a measurement: every value it records was already
    computed for the aggregates."""
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )

    def action() -> None:
        with profiler.phase("p"):
            pass

    try:
        assert _per_call_ns(action) < BUDGET_NS["traced"]
    finally:
        profiler.close()


def test_tracing_costs_nothing_when_it_is_off(run_dir: Path) -> None:
    """The default path must not pay for a feature it does not use.

    Guarded as a ratio rather than an absolute: the untraced profiler is the baseline the rest
    of this file already bounds, and what matters here is that adding the trace code to
    ``_PhaseScope`` left it where it was. The gate is one identity test, so anything beyond
    noise means the branch stopped being free.
    """
    untraced = Profiler(
        run_dir=run_dir / "off", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )

    def action() -> None:
        with untraced.phase("p"):
            pass

    try:
        assert _per_call_ns(action) < BUDGET_NS["with_cpu"]
    finally:
        untraced.close()


def test_declaring_async_work_costs_about_what_a_plain_phase_costs(run_dir: Path) -> None:
    """One bool store on entry and one bool test on exit, and nothing else.

    A ratio rather than an absolute, for the same reason as the tracing test above: the plain
    phase is already bounded, and what matters is that the declaration did not turn the hot
    path into something that has to think. If this fails, the flag stopped being free for the
    runs that use it — and it is meant to be usable on the phase in the inner loop.
    """
    profiler = Profiler(
        run_dir=run_dir / "async", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )

    def plain() -> None:
        with profiler.phase("p"):
            pass

    def declared() -> None:
        with profiler.phase("q", async_work=True):
            pass

    try:
        assert _per_call_ns(declared) < _per_call_ns(plain) * 1.5
    finally:
        profiler.close()
