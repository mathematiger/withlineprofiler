"""``import with_line_profiler`` — the pip name, for people who tried it first.

The distribution installs as ``with-line-profiler`` and imports as ``lineprofiler``, and the
mismatch costs a first-time reader a few minutes and an ``ImportError`` that says nothing
about the real name. This module is that redirection and nothing else: ``lineprofiler``
remains the documented name, and no feature will ever live only here.

    import with_line_profiler as lp

    with lp.LineProfiler() as profiler:
        ...
"""

from lineprofiler import (
    FunctionStats,
    LineProfiler,
    LineStats,
    __version__,
    accounting,
    start_profiling,
    stop_profiling,
)

__all__ = [
    "FunctionStats",
    "LineProfiler",
    "LineStats",
    "__version__",
    "accounting",
    "start_profiling",
    "stop_profiling",
]
