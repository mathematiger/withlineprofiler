"""What a line event costs, per engine, against ``line_profiler`` used directly.

The numbers quoted in docs/comparison.md come from here. Re-run and update them if the hot
path changes:

    poetry run python benchmarks/bench_lineprofiler.py

One machine, one workload; good to a factor of about 1.5, which is the resolution the
comparison table claims. What it is really for is the *ratio* between the two engines.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import line_profiler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_workload import work  # noqa: E402

ITERATIONS = 300_000
EVENTS = ITERATIONS * 2  # the loop's two lines
REPEATS = 3


def _best(run: object) -> float:
    """The fastest of ``REPEATS`` runs: the least noise, not the average of it."""
    return min(_timed(run) for _ in range(REPEATS))  # type: ignore[arg-type]


def _timed(run: object) -> float:
    start = time.perf_counter()
    run()  # type: ignore[operator]
    return time.perf_counter() - start


def _row(label: str, seconds: float, baseline: float) -> str:
    per_event = seconds / EVENTS * 1e9
    return f"{label:<34} {seconds * 1e3:8.1f} ms {per_event:8.0f} ns {seconds / baseline:7.1f}x"


def main() -> None:
    from lineprofiler import LineProfiler

    here = str(Path(__file__).resolve().parent)
    baseline = _best(lambda: work(ITERATIONS))

    print(f"{'':<34} {'runtime':>11} {'per event':>11} {'vs base':>8}")  # noqa: T201
    print(_row("no profiler", baseline, baseline))  # noqa: T201

    engines = (("line_profiler", None), ("builtin", "monitoring"), ("builtin", "settrace"))
    for engine, backend in engines:
        if backend == "monitoring" and not hasattr(sys, "monitoring"):
            continue

        def run(engine: str = engine, backend: str | None = backend) -> None:
            profiler = LineProfiler(project_folder=here, engine=engine, backend=backend)  # type: ignore[arg-type]
            with profiler:
                work(ITERATIONS)

        label = engine if backend is None else f"{engine} ({backend})"
        print(_row(label, _best(run), baseline))  # noqa: T201

    # Registered once, outside the timing: add_function pads the function's bytecode, and
    # registering it per repeat would accumulate padding and time a slower function each time.
    direct = line_profiler.LineProfiler(work)

    def raw() -> None:
        direct.runcall(work, ITERATIONS)

    print(_row("line_profiler, called directly", _best(raw), baseline))  # noqa: T201
    print(f"\nbest of {REPEATS} runs; {EVENTS:,} line events per run")  # noqa: T201
    _regions(here, baseline)


def _regions(here: str, baseline: float) -> None:  # noqa: ARG001
    """What a region costs: per line event while one is open, and per boundary crossed.

    The two columns are measured differently on purpose. *Open, per event* profiles the real
    workload with a region wrapped around it, so it is the cost of billing each line twice.
    *Per boundary* points the profiler at an empty directory, so neither the timing loop nor
    the region's own machinery is traced and what is left is the switch itself; the cost of
    the ``with`` line you write is one line event, already priced in the other column.

    The engines trade places here, which is the point of measuring it. The C engine keeps a
    second profiler per region and pays to switch it on and off. The pure-Python engine only
    appends to a list at the boundary, and bills every open region on every line instead.
    """
    from lineprofiler import LineProfiler

    print(f"\n{'':<34} {'open, per event':>16} {'per boundary':>14}")  # noqa: T201
    engines = (("line_profiler", None), ("builtin", "monitoring"), ("builtin", "settrace"))
    for engine, backend in engines:
        if backend == "monitoring" and not hasattr(sys, "monitoring"):
            continue

        def inside(engine: str = engine, backend: str | None = backend) -> None:
            profiler = LineProfiler(project_folder=here, engine=engine, backend=backend)  # type: ignore[arg-type]
            with profiler, profiler.region("all"):
                work(ITERATIONS)

        label = engine if backend is None else f"{engine} ({backend})"
        per_event = _best(inside) / EVENTS * 1e9
        per_boundary = min(_switch_cost(engine, backend) for _ in range(REPEATS)) * 1e9
        print(f"{label:<34} {per_event:13.0f} ns {per_boundary:11.0f} ns")  # noqa: T201


BOUNDARIES = 20_000


def _switch_cost(engine: str, backend: str | None) -> float:
    """Seconds to enter and leave a region, with nothing else being profiled."""
    from lineprofiler import LineProfiler

    with tempfile.TemporaryDirectory() as empty:
        profiler = LineProfiler(project_folder=empty, engine=engine, backend=backend)  # type: ignore[arg-type]
        with profiler:
            with profiler.region("b"):  # bind the region before it is timed
                pass
            start = time.perf_counter()
            for _ in range(BOUNDARIES):
                with profiler.region("b"):
                    pass
            return (time.perf_counter() - start) / BOUNDARIES


if __name__ == "__main__":
    main()
