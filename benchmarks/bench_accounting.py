"""Measure the per-phase cost of the accounting layer.

Run: ``python benchmarks/bench_accounting.py``

The numbers this prints are the ones quoted in the README. They decide where instrumentation
belongs: a phase costs roughly a microsecond, so phases go around MCTS searches and env
steps, and ``count()`` goes inside the simulation loop.
"""

from __future__ import annotations

import tempfile
import time

from lineprofiler import accounting
from lineprofiler.accounting import Profiler

ITERATIONS = 200_000


def measure(label: str, action: object, iterations: int = ITERATIONS) -> float:
    """Time ``action`` and print nanoseconds per call."""
    call = action  # type: ignore[assignment]
    call()  # type: ignore[operator]
    start = time.perf_counter_ns()
    for _ in range(iterations):
        call()  # type: ignore[operator]
    per_call = (time.perf_counter_ns() - start) / iterations
    print(f"{label:<48}{per_call:8.1f} ns/call")  # noqa: T201
    return per_call


def main() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        disabled = Profiler(run_dir=run_dir, enabled=False)
        with_cpu = Profiler(run_dir=run_dir + "/on", enabled=True, snapshot_interval_s=None)
        no_cpu = Profiler(
            run_dir=run_dir + "/nocpu",
            enabled=True,
            snapshot_interval_s=None,
            measure_cpu=False,
        )
        traced = Profiler(
            run_dir=run_dir + "/traced",
            enabled=True,
            snapshot_interval_s=None,
            trace=True,
        )

        def empty() -> None:
            pass

        def phase_disabled() -> None:
            with disabled.phase("p"):
                pass

        def phase_no_cpu() -> None:
            with no_cpu.phase("p"):
                pass

        def phase_with_cpu() -> None:
            with with_cpu.phase("p"):
                pass

        def phase_with_io() -> None:
            with with_cpu.phase("p", io=True):
                pass

        def count_only() -> None:
            with_cpu.count("units")

        def phase_sampled() -> None:
            with with_cpu.phase("sampled", sample=0.01):
                pass

        def phase_ambient_uninstalled() -> None:
            with accounting.phase("p"):
                pass

        def phase_traced() -> None:
            with traced.phase("p"):
                pass

        def signal_only() -> None:
            traced.signal("batch", 1)

        def phase_async() -> None:
            with with_cpu.phase("p", async_work=True):
                pass

        def trace_mark_only() -> None:
            traced.trace_mark("inference", 1, "admitted")

        print(f"\n=== accounting overhead ({ITERATIONS:,} iterations) ===")  # noqa: T201
        baseline = measure("baseline: empty function call", empty)
        measure("phase(), enabled=False", phase_disabled)
        measure("phase(), enabled=True, measure_cpu=False", phase_no_cpu)
        measure("phase(), enabled=True, measure_cpu=True", phase_with_cpu)
        # Two /proc reads per end, so this runs at a fraction of the iteration count.
        measure("phase(io=True)", phase_with_io, iterations=ITERATIONS // 20)
        measure("count()", count_only)
        measure("phase(sample=0.01), skipped entry", phase_sampled)
        # The timeline's cost, and the reason it is off by default. The untraced rows above
        # are what a default profiler pays; this is what turning it on adds.
        measure("phase(), trace=True", phase_traced)
        measure("phase(async_work=True)", phase_async)
        measure("signal()", signal_only)
        measure("trace_mark()", trace_mark_only)
        accounting.uninstall_profiler()
        measure("accounting.phase(), nothing installed", phase_ambient_uninstalled)
        print(f"\n(baseline call overhead of {baseline:.0f} ns is included in every row)")  # noqa: T201

        _report_in_phase_floor(
            ("measure_cpu=False", no_cpu),
            ("measure_cpu=True", with_cpu),
            ("measure_cpu=True, trace=True", traced),
        )

        for profiler in (with_cpu, no_cpu, traced):
            profiler.close()


def _report_in_phase_floor(*profilers: tuple[str, Profiler]) -> None:
    """Print what each profiler bills *into* a phase, as opposed to beside it.

    A different quantity from every row above, and the one that bounds accuracy rather than
    speed: ``__enter__`` reads its clock last and ``__exit__`` reads its first, so most of the
    cost above falls outside the measured interval. What is left inside inflates the phase's
    own reported wall time, once per entry, regardless of how long the phase ran.

    Read straight off each profiler's own aggregate for an empty body, which is exactly what
    a user would see for a phase that did nothing.
    """
    print("\n=== billed inside the phase (inflates its reported wall time) ===")  # noqa: T201
    for label, profiler in profilers:
        stats = profiler.merged_tree().get(("p",))
        if stats is None or stats.calls == 0:
            continue
        print(f"{label:<48}{stats.wall_ns / stats.calls:8.1f} ns/phase")  # noqa: T201
    print(  # noqa: T201
        "\n(an empty phase reports this much; a phase shorter than ~10 us is measurably\n"
        " inflated by it, and docs/accounting-recipes.md gives the ratios)",
    )


if __name__ == "__main__":
    main()
