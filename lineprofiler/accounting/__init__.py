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

Afterwards, read the run with ``lineprofiler report profile/`` (or ``python -m lineprofiler
report profile/``), or from the script that produced it:

    from lineprofiler.accounting import write_report

    write_report("profile", "report.html", format="html")
"""

import json
from pathlib import Path

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
    signal_ready,
    start,
    stop,
    trace_begin,
    trace_end,
    trace_mark,
    uninstall_profiler,
    wait_on,
)
from lineprofiler.accounting.report import render, report_as_dict
from lineprofiler.accounting.sampler import ResourceSampler, Sample
from lineprofiler.accounting.snapshot import MergedRun, WorkerSnapshot, merge_run
from lineprofiler.accounting.trace import Span, TraceBuffer, WorkerTrace
from lineprofiler.accounting.tracealign import lifecycle_segments, overlap_ns


def write_report(run_dir: str | Path, path: str | Path, format: str = "text") -> None:  # noqa: A002
    """Merge ``run_dir`` and write its report to ``path``. The library face of ``report``.

    The path from a run directory to a file someone can open used to run through the command
    line only, so a training script that had just finished a run could not save its own report
    without shelling out to itself. Formats are exactly the CLI's — ``"text"``, ``"json"``,
    ``"html"`` — so there is one story about what a report can be, and an unrecognised one
    raises rather than quietly falling back to text.

    Parent directories are created, as :func:`~lineprofiler.accounting.htmlreport.write_html`
    already does: a caller writing to ``reports/run-17.html`` from code means it. The CLI
    deliberately does the opposite, where a path that does not exist is usually a typo.

    Test specifically:
        - each format writes a non-empty file, and json parses
        - a nested destination directory is created
        - an unknown format raises rather than writing something unexpected
    """
    run = merge_run(run_dir, with_trace=True)
    if format == "json":
        _write_text(path, json.dumps(report_as_dict(run), indent=2))
        return
    if format == "html":
        from lineprofiler.accounting.htmlreport import write_html

        write_html(run, path)
        return
    if format != "text":
        raise ValueError(f"format must be 'text', 'json' or 'html', got {format!r}")
    _write_text(path, render(run))


def write_trace(run_dir: str | Path, path: str | Path, format: str = "html") -> None:  # noqa: A002
    """Merge ``run_dir`` and write its timeline to ``path``. The library face of ``trace``.

    ``"html"`` first for the same reason the CLI defaults to it: a timeline is a picture, and
    the text form of one is a wall of numbers. ``"json"`` emits the document
    :func:`~lineprofiler.accounting.findings.trace_as_dict` builds, which is the one the CLI
    emits too.

    Test specifically:
        - html and json both write a file; an unknown format raises
        - the json is identical to what ``lineprofiler trace --format json`` prints
    """
    if format not in {"html", "json"}:
        raise ValueError(f"format must be 'html' or 'json', got {format!r}")
    # Merged with the trace, always: every conclusion this writes — the findings, the
    # occupancy, the lifecycle — is derived from spans, and a timeline without them is a page
    # explaining how to record one.
    run = merge_run(run_dir, with_trace=True)
    if format == "json":
        from lineprofiler.accounting.findings import trace_as_dict
        from lineprofiler.accounting.tracealign import align_run

        _write_text(path, json.dumps(trace_as_dict(align_run(run)), indent=2))
        return
    from lineprofiler.accounting.htmltrace import write_trace_html

    write_trace_html(run, path)


def _write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


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
    "Span",
    "TraceBuffer",
    "WorkerSnapshot",
    "WorkerTrace",
    "analyse",
    "compare",
    "count",
    "current",
    "install_profiler",
    "installed_profiler",
    "merge_run",
    "lifecycle_segments",
    "merge_trees",
    "overlap_ns",
    "phase",
    "render",
    "render_comparison",
    "report_as_dict",
    "signal_ready",
    "start",
    "stop",
    "trace_begin",
    "trace_end",
    "trace_mark",
    "uninstall_profiler",
    "wait_on",
    "write_report",
    "write_trace",
]
