"""Context-manager based line-by-line profiler for Python functions.

This module provides a simple context manager interface for profiling code blocks
and functions with detailed line-by-line timing information.
"""
from __future__ import annotations

import inspect
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType, TracebackType
from typing import Optional

from lineprofiler.config import ProfilerConfig, find_project_root, get_config

# Signature of a CPython trace function as installed via sys.settrace.
TraceFunction = Callable[[FrameType, str, object], Optional["TraceFunction"]]


def _qualname_of(code: CodeType) -> str:
    """Return ``co_qualname`` where the interpreter has it (3.11+), else ``co_name``.

    On 3.10 a method or nested function is matched by its bare name, so a
    ``functions = ["MyClass.step"]`` pattern will not match there. That limitation is
    documented rather than emulated: reconstructing a qualified name from a code object
    needs the enclosing frame chain, and a *guessed* qualname would silently match the
    wrong function — a wrong answer where this returns an honestly narrower one.
    """
    return getattr(code, "co_qualname", code.co_name)

# A function is identified by the file it lives in, its name and its first line.
FunctionKey = tuple[str, str, int]


@dataclass
class LineStats:
    """Statistics for a single line of code."""

    line_number: int
    hits: int = 0
    total_time: float = 0.0

    @property
    def average_time(self) -> float:
        """Average wall-clock time per execution, or 0.0 if never hit."""
        return self.total_time / self.hits if self.hits > 0 else 0.0


@dataclass
class FunctionStats:
    """Accumulated statistics for an entire function.

    ``source_lines`` is shared (by reference) between all functions that live in
    the same file, so the source of a file is held in memory only once.
    """

    filename: str
    function_name: str
    first_line: int
    line_stats: dict[int, LineStats] = field(default_factory=dict)
    source_lines: dict[int, str] = field(default_factory=dict)
    total_time: float = 0.0


class LineProfiler:
    """Context manager for line-by-line profiling of code blocks.

    Usage:
        profiler = LineProfiler()
        with profiler:
            result = some_function()
        profiler.print_stats()

    Timing model
    ------------
    The interpreter calls the trace callback before every traced line. The time
    between two consecutive trace events is attributed to the line that was
    *about to finish* (the previously executing line). The reference timestamp
    for the next line is taken at the very end of the callback, so the profiler's
    own bookkeeping is excluded from the reported per-line times.

    Only frames whose file lives inside ``project_folder`` are traced. Calls into
    the standard library or third-party packages are skipped, which keeps both the
    overhead and the output focused on the user's own code.
    """

    def __init__(
        self,
        project_folder: str | Path | None = None,
        config: ProfilerConfig | None = None,
    ) -> None:
        """Initialize the profiler.

        Args:
            project_folder: Folder used to scope profiling. If omitted, it is
                auto-detected by walking up from the caller's file to the nearest
                git repository root.
            config: Optional include/exclude/function glob filters, on top of
                ``project_folder``. Defaults to no extra filtering (everything under
                ``project_folder`` is traced), which is the pre-existing behavior.
        """
        self._function_stats: dict[FunctionKey, FunctionStats] = {}
        self._enabled: bool = False
        self._last_time: float = 0.0
        self._last_line: int | None = None
        self._last_key: FunctionKey | None = None
        self._project_cache: dict[str, bool] = {}
        self._source_cache: dict[str, dict[int, str]] = {}
        self._old_trace = sys.gettrace()
        self._config = config

        if project_folder is not None:
            self._project_folder: Path = Path(project_folder).resolve()
        else:
            caller_frame = inspect.currentframe()
            caller = caller_frame.f_back if caller_frame else None
            if caller is not None:
                self._project_folder = self._find_repo_root(caller.f_code.co_filename)
            else:
                self._project_folder = Path.cwd()

    def __enter__(self) -> LineProfiler:
        """Enable profiling, registering the trace callback.

        Re-entering an instance that is already active is refused rather than allowed to
        corrupt it. The nested ``__enter__`` used to save *this profiler's own callback* as
        the tracer to restore, so the outer ``__exit__`` reinstalled the profiler instead of
        clearing it — leaving a global trace function dispatched on every Python call for the
        rest of the process, invisibly, since ``_enabled`` was by then ``False``.

        Note that the body of the ``with`` block is not itself profiled: ``sys.settrace``
        only affects frames created after it is installed, so only functions *called* from
        the block appear. Put the code you want measured in a function.
        """
        if self._enabled:
            raise RuntimeError(
                "this LineProfiler is already active; nesting the same instance would leak "
                "its trace function for the lifetime of the process. Use a second instance.",
            )
        self._enabled = True
        self._old_trace = sys.gettrace()
        self._last_key = None
        self._last_line = None
        sys.settrace(self._trace_callback)
        self._last_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disable profiling and restore the previous trace function."""
        self._enabled = False
        sys.settrace(self._old_trace)
        self._old_trace = None

    def _trace_callback(
        self,
        frame: FrameType,
        event: str,
        arg: object,  # noqa: ARG002
    ) -> TraceFunction | None:
        """Trace callback invoked by the interpreter for each traced event.

        Returns the callback itself to keep tracing a frame, or ``None`` to skip
        a frame that is outside the project folder.
        """
        if not self._enabled:
            return None

        now = time.perf_counter()

        if event == "line":
            self._record_gap(now)
            self._last_key = self._ensure_function(frame)
            self._last_line = frame.f_lineno
            self._last_time = time.perf_counter()
            return self._trace_callback

        if event == "call":
            if not self._admits(frame.f_code):
                return None
            self._record_gap(now)
            self._ensure_function(frame)
            self._last_key = None
            self._last_line = None
            self._last_time = time.perf_counter()
            return self._trace_callback

        if event == "return":
            self._record_gap(now)
            self._last_key = None
            self._last_line = None
            self._last_time = time.perf_counter()

        return self._trace_callback

    def _record_gap(self, now: float) -> None:
        """Attribute the elapsed time since the last event to the last line."""
        if self._last_key is None or self._last_line is None:
            return

        func_stats = self._function_stats[self._last_key]
        line_stats = func_stats.line_stats.get(self._last_line)
        if line_stats is None:
            line_stats = LineStats(line_number=self._last_line)
            func_stats.line_stats[self._last_line] = line_stats

        delta = now - self._last_time
        line_stats.hits += 1
        line_stats.total_time += delta
        func_stats.total_time += delta

    def _admits(self, code: CodeType) -> bool:
        """Whether ``code`` is inside the project folder and passes the configured filters.

        The single admission decision, so the project-folder check and the function-name
        glob cannot drift apart between the places that ask.
        """
        if not self._is_in_project_folder(code.co_filename):
            return False
        return self._config is None or self._config.allows_function(_qualname_of(code))

    def _ensure_function(self, frame: FrameType) -> FunctionKey:
        """Return the key for ``frame``'s function, creating its stats on demand."""
        return self._ensure_function_of(frame.f_code)

    def _ensure_function_of(self, code: CodeType) -> FunctionKey:
        """Return the key for ``code``'s function, creating its stats on demand."""
        key: FunctionKey = (code.co_filename, code.co_name, code.co_firstlineno)
        if key not in self._function_stats:
            self._function_stats[key] = FunctionStats(
                filename=code.co_filename,
                function_name=code.co_name,
                first_line=code.co_firstlineno,
                source_lines=self._get_file_lines(code.co_filename),
            )
        return key

    def _get_file_lines(self, filename: str) -> dict[int, str]:
        """Read and cache the source lines of ``filename`` (once per file)."""
        cached = self._source_cache.get(filename)
        if cached is not None:
            return cached

        lines: dict[int, str] = {}
        try:
            path = Path(filename)
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    for i, line in enumerate(f, start=1):
                        lines[i] = line.rstrip()
        except (OSError, UnicodeDecodeError):
            # If we can't read the file, just continue with no source.
            pass

        self._source_cache[filename] = lines
        return lines

    def _find_repo_root(self, start_path: str) -> Path:
        """Return the git repo root (directory containing .git), else the caller's directory.

        The fallback returns the *directory*, not the file. Returning the file path meant
        ``relative_to`` matched only that one module, so outside a git checkout — a
        pip-installed application, an sdist, a container built without ``.git`` — the
        profiler silently narrowed to the single file that had constructed it, and reported
        nothing about the rest of the project.
        """
        return find_project_root(start_path)

    def _is_in_project_folder(self, filename: str) -> bool:
        """Return whether ``filename`` lives inside the project folder (cached).

        When a ``config`` was passed to the constructor, its ``include``/``exclude`` globs
        (evaluated against the path relative to the project folder) narrow this further.
        """
        cached = self._project_cache.get(filename)
        if cached is not None:
            return cached

        try:
            relative = Path(filename).resolve().relative_to(self._project_folder)
            result = self._config is None or self._config.allows_path(str(relative))
        except (OSError, ValueError):
            result = False

        self._project_cache[filename] = result
        return result

    def print_stats(  # noqa: C901
        self,
        min_time_us: float = 0.0,
        top_n_lines: int | None = None,
        sort_by: str = "time",
    ) -> None:
        """Print detailed profiling statistics per function.

        Args:
            min_time_us: Minimum time in microseconds to display a line.
            top_n_lines: If set, only show the top N lines per function.
            sort_by: How to sort lines - "time" (total time), "hits" (call count),
                    or "line" (line number). Default is "time".
        """
        if not self._function_stats:
            print("No profiling data collected.")  # noqa: T201
            return

        for key, func_stats in sorted(self._function_stats.items()):
            filename, function_name, first_line = key

            if not self._is_in_project_folder(filename):
                print(f"filename not in folde: {filename}")  # noqa: T201
                continue

            if not func_stats.line_stats:
                continue

            print("=" * 100)  # noqa: T201
            print(f"File: {filename}")  # noqa: T201
            print(f"Function: {function_name} at line {first_line}")  # noqa: T201
            print(f"Total time: {func_stats.total_time * 1e6:.1f} µs")  # noqa: T201
            print("=" * 100)  # noqa: T201
            header = (
                f"{'Line #':<8} {'Hits':<10} {'Time (µs)':<15} "
                f"{'Per Hit (µs)':<15} {'% Time':<10} {'Line Content'}"
            )
            print(header)  # noqa: T201
            print("-" * 100)  # noqa: T201

            line_data = self._select_lines(func_stats, min_time_us, top_n_lines, sort_by)

            for line_num, line_stats in line_data:
                self._print_line_row(line_num, line_stats, func_stats)

            print()  # noqa: T201

    def _select_lines(
        self,
        func_stats: FunctionStats,
        min_time_us: float,
        top_n_lines: int | None,
        sort_by: str,
    ) -> list[tuple[int, LineStats]]:
        """Filter, sort and limit a function's lines for display."""
        line_data = [
            (line_num, line_stats)
            for line_num, line_stats in func_stats.line_stats.items()
            if line_stats.total_time * 1e6 >= min_time_us
        ]

        if sort_by == "time":
            line_data.sort(key=lambda x: x[1].total_time, reverse=True)
        elif sort_by == "hits":
            line_data.sort(key=lambda x: x[1].hits, reverse=True)
        else:  # sort_by == "line"
            line_data.sort(key=lambda x: x[0])

        if top_n_lines is not None:
            line_data = line_data[:top_n_lines]

        return line_data

    def _print_line_row(
        self,
        line_num: int,
        line_stats: LineStats,
        func_stats: FunctionStats,
    ) -> None:
        """Print a single line row of a per-function table."""
        time_us = line_stats.total_time * 1e6
        avg_time_us = line_stats.average_time * 1e6
        percent = (
            line_stats.total_time / func_stats.total_time * 100
            if func_stats.total_time > 0
            else 0.0
        )

        source_line = func_stats.source_lines.get(line_num, "")
        if len(source_line) > 50:  # noqa: PLR2004
            source_line = source_line[:47] + "..."

        print(  # noqa: T201
            f"{line_num:<8} {line_stats.hits:<10} {time_us:<15.1f} "
            f"{avg_time_us:<15.1f} {percent:<10.1f} {source_line}"
        )

    def print_global_top_stats(
        self,
        top_n: int = 10,
        min_time_us: float = 0.0,
        sort_by: str = "time",
    ) -> None:
        """Print a global summary of the top lines across all functions.

        Args:
            top_n: Number of top lines to display.
            min_time_us: Minimum time in microseconds to include a line.
            sort_by: How to sort - "time" (total time) or "hits" (call count).
        """
        all_lines = self._collect_global_lines(min_time_us)

        if not all_lines:
            print("No profiling data above the threshold.")  # noqa: T201
            return

        if sort_by == "hits":
            all_lines.sort(key=lambda e: e.hits, reverse=True)
        else:  # sort_by == "time"
            all_lines.sort(key=lambda e: e.time_us, reverse=True)

        print("=" * 130)  # noqa: T201
        print(f"Top {top_n} lines across all functions (sorted by {sort_by})")  # noqa: T201
        print("=" * 130)  # noqa: T201
        header = (
            f"{'File::Function':<50} {'Line':<6} {'Hits':<10} {'Time (µs)':<13} "
            f"{'Per Hit (µs)':<14} {'% Time':<8} {'Line Content'}"
        )
        print(header)  # noqa: T201
        print("-" * 130)  # noqa: T201

        for entry in all_lines[:top_n]:
            self._print_global_row(entry)

        print("=" * 130)  # noqa: T201
        print()  # noqa: T201

    def _collect_global_lines(self, min_time_us: float) -> list[_GlobalLine]:
        """Flatten every traced line into a list of global summary entries."""
        all_lines: list[_GlobalLine] = []

        for key, func_stats in self._function_stats.items():
            filename, function_name, _ = key

            if not self._is_in_project_folder(filename) or not func_stats.line_stats:
                continue

            short_filename = self._display_filename(filename)

            for line_num, line_stats in func_stats.line_stats.items():
                time_us = line_stats.total_time * 1e6
                if time_us < min_time_us:
                    continue

                percent = (
                    line_stats.total_time / func_stats.total_time * 100
                    if func_stats.total_time > 0
                    else 0.0
                )
                all_lines.append(
                    _GlobalLine(
                        file=short_filename,
                        function=function_name,
                        line_num=line_num,
                        hits=line_stats.hits,
                        time_us=time_us,
                        avg_time_us=line_stats.average_time * 1e6,
                        percent=percent,
                        source_line=func_stats.source_lines.get(line_num, ""),
                    )
                )

        return all_lines

    def _display_filename(self, filename: str) -> str:
        """Return ``filename`` relative to the project folder, else its basename."""
        try:
            return str(Path(filename).resolve().relative_to(self._project_folder))
        except (ValueError, OSError):
            return Path(filename).name

    def _print_global_row(self, entry: _GlobalLine) -> None:
        """Print a single line row of the global summary table."""
        source_line = entry.source_line
        if len(source_line) > 40:  # noqa: PLR2004
            source_line = source_line[:37] + "..."

        file_func = f"{entry.file}::{entry.function}"
        if len(file_func) > 50:  # noqa: PLR2004
            file_func = file_func[:47] + "..."

        print(  # noqa: T201
            f"{file_func:<50} {entry.line_num:<6} {entry.hits:<10} "
            f"{entry.time_us:<13.1f} {entry.avg_time_us:<14.1f} "
            f"{entry.percent:<8.1f} {source_line}"
        )

    def get_stats(self) -> dict[FunctionKey, FunctionStats]:
        """Return the raw profiling statistics (a live reference, not a copy)."""
        return self._function_stats

    def clear(self) -> None:
        """Clear all profiling data and reset the timing state."""
        self._function_stats.clear()
        self._project_cache.clear()
        self._source_cache.clear()
        self._last_time = 0.0
        self._last_line = None
        self._last_key = None

    def reset(self) -> None:
        """Reset the profiler to its initial state (alias for ``clear``)."""
        self.clear()


@dataclass
class _GlobalLine:
    """A single line entry in the global cross-function summary."""

    file: str
    function: str
    line_num: int
    hits: int
    time_us: float
    avg_time_us: float
    percent: float
    source_line: str


# ── ambient profiling: start_profiling() / stop_profiling() ────────────────
#
# A `with` block cannot wrap a whole module or script, and `sys.settrace` only affects frames
# created after it is installed anyway. The two-line alternative below is opt-in
# (`LINEPROFILER_ENABLED`, see `lineprofiler.config`) and safe to leave in place permanently:
# with profiling disabled, `start_profiling()`/`stop_profiling()` cost one dict lookup each.

_installed: LineProfiler | None = None


def start_profiling(project_folder: str | Path | None = None) -> LineProfiler:
    """Start ambient line-by-line profiling — the two-line alternative to ``with profiler:``.

        from lineprofiler import start_profiling, stop_profiling

        start_profiling()      # top of the region/script
        ...
        stop_profiling()       # bottom of the region/script

    Opt-in: profiling only actually starts when ``LINEPROFILER_ENABLED`` is truthy or a
    ``[tool.lineprofiler]`` table exists (see ``lineprofiler.config.get_config``). Otherwise
    this returns a fresh, never-entered ``LineProfiler`` and installs nothing, so the call is
    always safe to leave in the code.

    Calling this again before ``stop_profiling()`` warns and returns the profiler already
    running, rather than raising — ambient usage should never crash the host program.
    """
    global _installed  # noqa: PLW0603 - the point of the function

    if _installed is not None:
        warnings.warn(
            "start_profiling() was already called; stop_profiling() first to start a new "
            "profiler. Returning the one already running.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _installed

    caller_frame = inspect.currentframe()
    caller = caller_frame.f_back if caller_frame else None
    start_path = project_folder if project_folder is not None else (
        caller.f_code.co_filename if caller is not None else Path.cwd()
    )
    config = get_config(start_path)

    resolved_folder = Path(project_folder) if project_folder is not None else find_project_root(
        start_path,
    )
    profiler = LineProfiler(project_folder=resolved_folder, config=config)
    if config.enabled:
        profiler.__enter__()
        _installed = profiler
    return profiler


def stop_profiling(print_stats: bool = True) -> LineProfiler | None:
    """Stop ambient profiling started by ``start_profiling()``.

    Returns the profiler so callers can still read ``get_stats()`` afterward, or ``None`` if
    ``start_profiling()`` was never called (or profiling was not enabled). Prints the top lines
    across all functions unless ``print_stats`` is ``False``.
    """
    global _installed  # noqa: PLW0603 - the point of the function

    profiler = _installed
    if profiler is None:
        return None

    profiler.__exit__(None, None, None)
    _installed = None
    if print_stats:
        profiler.print_global_top_stats()
    return profiler
