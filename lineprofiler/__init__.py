"""Line-by-line profiler package for Python.

This package provides a simple context manager interface for profiling code blocks
and functions with detailed line-by-line timing information.
"""

from lineprofiler.profiler import (
    FunctionStats,
    LineProfiler,
    LineStats,
)

__all__ = [
    "FunctionStats",
    "LineProfiler",
    "LineStats",
]

__version__ = "0.1.2"
