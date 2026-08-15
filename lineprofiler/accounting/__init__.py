"""Semantic accounting layer for long-running, multi-process RL training.

Records aggregates for regions you name, cheaply enough to leave on for a twelve-hour run,
across every process in a pipeline. It deliberately does not trace functions, time CUDA
kernels or attribute memory per line — ``torch.profiler``, VizTracer, memray and nsys
already do those, and the ``backend`` option starts them for a bounded window instead of
reimplementing them.

    from lineprofiler.accounting import Profiler

    profiler = Profiler(run_dir="profile", role="actor")
    with profiler, profiler.phase("iteration"):
        ...
"""

from lineprofiler.accounting.analysis import SampleAnalysis, analyse
from lineprofiler.accounting.backend import Backend
from lineprofiler.accounting.compare import PhaseDelta, compare, render_comparison
from lineprofiler.accounting.histogram import DurationHistogram
from lineprofiler.accounting.phasetree import PhaseStats, PhaseTree, merge_trees
from lineprofiler.accounting.profiler import (
    Profiler,
    count,
    current,
    install_profiler,
    installed_profiler,
    phase,
    start,
    stop,
    uninstall_profiler,
)
from lineprofiler.accounting.report import render, report_as_dict
from lineprofiler.accounting.sampler import ResourceSampler, Sample
from lineprofiler.accounting.snapshot import MergedRun, WorkerSnapshot, merge_run

__all__ = [
    "Backend",
    "DurationHistogram",
    "MergedRun",
    "PhaseDelta",
    "PhaseStats",
    "PhaseTree",
    "Profiler",
    "ResourceSampler",
    "Sample",
    "SampleAnalysis",
    "WorkerSnapshot",
    "analyse",
    "compare",
    "count",
    "current",
    "install_profiler",
    "installed_profiler",
    "merge_run",
    "merge_trees",
    "phase",
    "render",
    "render_comparison",
    "report_as_dict",
    "start",
    "stop",
    "uninstall_profiler",
]
