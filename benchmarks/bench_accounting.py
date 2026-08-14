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

        print(f"\n=== accounting overhead ({ITERATIONS:,} iterations) ===")  # noqa: T201
        baseline = measure("baseline: empty function call", empty)
        measure("phase(), enabled=False", phase_disabled)
        measure("phase(), enabled=True, measure_cpu=False", phase_no_cpu)
        measure("phase(), enabled=True, measure_cpu=True", phase_with_cpu)
        # Two /proc reads per end, so this runs at a fraction of the iteration count.
        measure("phase(io=True)", phase_with_io, iterations=ITERATIONS // 20)
        measure("count()", count_only)
        measure("phase(sample=0.01), skipped entry", phase_sampled)
        accounting.uninstall_profiler()
        measure("accounting.phase(), nothing installed", phase_ambient_uninstalled)
        print(f"\n(baseline call overhead of {baseline:.0f} ns is included in every row)")  # noqa: T201

        for profiler in (with_cpu, no_cpu):
            profiler.close()


if __name__ == "__main__":
    main()
