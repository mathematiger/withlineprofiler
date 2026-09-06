"""Context-manager based line-by-line profiler for Python functions.

Two engines share one API. ``line_profiler`` — the default wherever it is installed — times
lines in C, keeps per-thread state and bills a call line inclusively. ``builtin`` is the
pure-Python fallback on ``sys.monitoring`` (3.12+) or ``sys.settrace``. Both are scoped to
the project folder, and both feed the same reports.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType, TracebackType
from typing import IO, TYPE_CHECKING, Literal, Optional

from lineprofiler.config import ProfilerConfig, find_project_root, get_config

if TYPE_CHECKING:
    from lineprofiler.engine_lp import LineProfilerEngine

# Signature of a CPython trace function as installed via sys.settrace.
TraceFunction = Callable[[FrameType, str, object], Optional["TraceFunction"]]

# A function is identified by the file it lives in, its name and its first line.
FunctionKey = tuple[str, str, int]

# Which interpreter facility delivers the builtin engine's events.
Backend = Literal["monitoring", "settrace"]

# Which code does the timing.
Engine = Literal["line_profiler", "builtin"]

# ``sys.monitoring`` arrived in 3.12. Resolved once here so the rest of the module can test
# for it with an ``is None`` rather than a version comparison per call.
_MONITORING = getattr(sys, "monitoring", None)

# sys.monitoring.PROFILER_ID. Spelled as a literal so this module still imports on 3.10/3.11,
# where the attribute does not exist. Coverage.py claims COVERAGE_ID (1), not this slot.
_TOOL_ID = 2


def _default_backend() -> Backend:
    """Return ``"monitoring"`` where the interpreter has it (3.12+), else ``"settrace"``."""
    return "monitoring" if _MONITORING is not None else "settrace"


def _default_engine() -> Engine:
    """Return ``"line_profiler"`` where the C engine is installed, else ``"builtin"``."""
    return "line_profiler" if importlib.util.find_spec("line_profiler") else "builtin"


def _qualname_of(code: CodeType) -> str:
    """Return ``co_qualname`` where the interpreter has it (3.11+), else ``co_name``.

    On 3.10 a method or nested function is matched by its bare name, so a
    ``functions = ["MyClass.step"]`` pattern will not match there. That limitation is
    documented rather than emulated: reconstructing a qualified name from a code object
    needs the enclosing frame chain, and a *guessed* qualname would silently match the
    wrong function — a wrong answer where this returns an honestly narrower one.
    """
    return getattr(code, "co_qualname", code.co_name)


def _key_of(code: CodeType) -> FunctionKey:
    return (code.co_filename, code.co_name, code.co_firstlineno)


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


class _Open:
    """One in-project frame on a thread's stack: which line is running, and since when."""

    __slots__ = ("key", "line", "started")

    def __init__(self, key: FunctionKey, started: float) -> None:
        self.key = key
        self.line: int | None = None
        self.started = started


class _FrameStack(threading.local):
    """The open in-project frames of one thread, innermost last.

    Thread-local because ``sys.monitoring`` delivers every thread's events to the same
    callbacks: one shared "last line" raced across threads and lost hits.
    """

    def __init__(self) -> None:
        self.entries: list[_Open] = []


class LineProfiler:
    """Context manager for line-by-line profiling of code blocks.

    Usage:
        profiler = LineProfiler()
        with profiler:
            result = some_function()
        profiler.print_stats()

    Only frames whose file lives inside ``project_folder`` are traced. Calls into the
    standard library or third-party packages are skipped, which keeps both the overhead and
    the output focused on the user's own code.

    Engines
    -------
    ``engine="line_profiler"`` (the default where the package is installed) hands the timing
    to ``line_profiler``'s C callback and only decides *what* it watches — every admitted
    function, discovered on first call on 3.12+ and up front below that. ``engine="builtin"``
    is pure Python, chosen automatically when ``line_profiler`` is absent or when ``backend=``
    is passed.

    Timing model (both engines)
    ---------------------------
    The time between two consecutive line events in a frame is billed to the line that was
    running, so a line that calls a function is billed the whole call, the same way
    ``line_profiler`` and every other line profiler do it. Each thread keeps its own stack of
    open frames, so threads do not corrupt each other's numbers.

    Builtin backends
    ----------------
    Events come from ``sys.monitoring`` on 3.12+ and ``sys.settrace`` below it, chosen
    automatically. ``sys.settrace`` is a single global hook, so that backend cannot run
    alongside coverage.py or pdb, and it only affects frames created *after* it is
    installed, so the body of the ``with`` block itself is not profiled — only functions
    called from it. ``sys.monitoring`` has neither restriction.
    """

    def __init__(
        self,
        project_folder: str | Path | None = None,
        config: ProfilerConfig | None = None,
        *,
        backend: Backend | None = None,
        engine: Engine | None = None,
    ) -> None:
        """Initialize the profiler.

        Args:
            project_folder: Folder used to scope profiling. If omitted, it is
                auto-detected by walking up from the caller's file to the nearest
                git repository root.
            config: Optional include/exclude/function glob filters, on top of
                ``project_folder``. Defaults to no extra filtering (everything under
                ``project_folder`` is traced).
            backend: The builtin engine's event source, ``"monitoring"`` or ``"settrace"``.
                Passing it selects the builtin engine. Passing ``"monitoring"`` below 3.12
                raises rather than quietly downgrading.
            engine: ``"line_profiler"`` or ``"builtin"``. Defaults to ``"line_profiler"``
                where it is installed, else ``"builtin"``.
        """
        if backend == "monitoring" and _MONITORING is None:
            raise ValueError(
                "the monitoring backend needs Python 3.12 or newer; this interpreter has "
                f"{sys.version_info.major}.{sys.version_info.minor}. Omit backend= to use "
                "sys.settrace here.",
            )
        if engine is None:
            engine = "builtin" if backend is not None else _default_engine()
        self._engine: Engine = engine
        self._backend: Backend = backend or _default_backend()
        self._function_stats: dict[FunctionKey, FunctionStats] = {}
        self._enabled: bool = False
        self._frames = _FrameStack()
        self._project_cache: dict[str, bool] = {}
        self._source_cache: dict[str, dict[int, str]] = {}
        self._old_trace = sys.gettrace()
        self._old_thread_trace: TraceFunction | None = None
        self._config = config
        self._lp: LineProfilerEngine | None = None
        self._region_stack: list[str] = []
        self._region_entries: dict[str, int] = {}
        self._region_stats: dict[str, dict[FunctionKey, FunctionStats]] = {}
        self._region_scopes: dict[str, _RegionScope] = {}

        if project_folder is not None:
            self._project_folder: Path = Path(project_folder).resolve()
        else:
            caller_frame = inspect.currentframe()
            caller = caller_frame.f_back if caller_frame else None
            if caller is not None:
                self._project_folder = self._find_repo_root(caller.f_code.co_filename)
            else:
                self._project_folder = Path.cwd()

    @property
    def backend(self) -> Backend:
        """The builtin engine's event source: ``"monitoring"`` or ``"settrace"``."""
        return self._backend

    @property
    def engine(self) -> Engine:
        """Which code does the timing: ``"line_profiler"`` or ``"builtin"``."""
        return self._engine

    def __enter__(self) -> LineProfiler:
        """Enable profiling, registering the event callbacks.

        Re-entering an instance that is already active is refused rather than allowed to
        corrupt it: the nested ``__enter__`` used to save *this profiler's own callback* as
        the tracer to restore, leaving a global trace function installed for the rest of the
        process.
        """
        if self._enabled:
            raise RuntimeError(
                "this LineProfiler is already active; nesting the same instance would leak "
                "its trace function for the lifetime of the process. Use a second instance.",
            )
        self._enabled = True
        self._frames = _FrameStack()
        if self._engine == "line_profiler":
            self._lp_engine().enable()
        elif self._backend == "monitoring":
            self._enable_monitoring()
        else:
            self._old_trace = sys.gettrace()
            self._old_thread_trace = threading.gettrace()
            sys.settrace(self._trace_callback)
            threading.settrace(self._trace_callback)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disable profiling and restore the interpreter to how it was found."""
        self._enabled = False
        if self._engine == "line_profiler":
            self._lp_engine().disable()
            self._sync_from_engine()
        elif self._backend == "monitoring":
            self._disable_monitoring()
        else:
            sys.settrace(self._old_trace)
            threading.settrace(self._old_thread_trace)
            self._old_trace = None
            self._old_thread_trace = None

    # ── the line_profiler engine ───────────────────────────────────────────

    def _lp_engine(self) -> LineProfilerEngine:
        if self._lp is None:
            from lineprofiler.engine_lp import LineProfilerEngine

            self._lp = LineProfilerEngine(self._admits, self._is_in_project_folder)
        return self._lp

    def _sync_from_engine(self) -> None:
        """Refresh ``_function_stats`` in place, so ``get_stats()`` keeps returning one dict."""
        if self._lp is None:
            return
        self._function_stats.clear()
        self._function_stats.update(self._lp.function_stats(self._get_file_lines))

    # ── the builtin engine: sys.monitoring ─────────────────────────────────

    def _enable_monitoring(self) -> None:
        """Claim the profiler tool slot, register callbacks, and clear any stale opt-outs.

        ``restart_events()`` is load-bearing, not hygiene. A code object this profiler
        returned ``DISABLE`` for in an *earlier* session stays disabled for the life of the
        interpreter, so a second ``with profiler:`` in the same process would record nothing
        for it and report a confident zero. It is called before ``set_events`` so no event
        can arrive between clearing the opt-outs and arming the callbacks.
        """
        mon = _MONITORING
        if mon is None:  # pragma: no cover - guarded by _backend, which the constructor sets
            raise RuntimeError("the monitoring backend is unavailable on this interpreter")
        try:
            mon.use_tool_id(_TOOL_ID, "lineprofiler")
        except ValueError as exc:
            raise RuntimeError(
                "another tool holds sys.monitoring's profiler slot, so this LineProfiler "
                "cannot start. Two line profilers would double-count every line; close the "
                "other one, or pass backend='settrace' to use the older hook instead.",
            ) from exc
        events = mon.events
        mon.register_callback(_TOOL_ID, events.LINE, self._on_line)
        for event in (events.PY_START, events.PY_RESUME):
            mon.register_callback(_TOOL_ID, event, self._on_start)
        for event in (events.PY_RETURN, events.PY_YIELD):
            mon.register_callback(_TOOL_ID, event, self._on_return)
        # The exceptional-path events cannot be locally disabled, so they get callbacks that
        # never return DISABLE. Returning it raises inside the interpreter's dispatch.
        mon.register_callback(_TOOL_ID, events.PY_THROW, self._on_throw)
        mon.register_callback(_TOOL_ID, events.PY_UNWIND, self._on_unwind)
        mon.restart_events()
        mon.set_events(_TOOL_ID, sum(_builtin_events(mon)))

    def _disable_monitoring(self) -> None:
        """Stop events and hand the tool slot back, so a later profiler can claim it."""
        mon = _MONITORING
        if mon is None:  # pragma: no cover - unreachable for the same reason as above
            return
        mon.set_events(_TOOL_ID, mon.events.NO_EVENTS)
        for event in _builtin_events(mon):
            mon.register_callback(_TOOL_ID, event, None)
        mon.free_tool_id(_TOOL_ID)

    def _on_line(self, code: CodeType, line_number: int) -> object:
        """``LINE``: bill the elapsed gap to this frame's previous line, then arm this one.

        The admission check cannot be left to ``PY_START`` alone. That event fires only for
        frames the interpreter *starts* while monitoring is armed, so any frame already on
        the stack at ``__enter__`` — this profiler's own ``__enter__`` among them — would
        otherwise have its remaining lines recorded unfiltered.
        """
        if not self._enabled or not self._admits(code):
            return _MONITORING.DISABLE  # type: ignore[union-attr]
        self._mark_line(code, line_number, time.perf_counter())
        return None

    def _on_start(self, code: CodeType, offset: int) -> object:  # noqa: ARG002
        """``PY_START`` / ``PY_RESUME`` / ``PY_THROW``: open a frame, or opt out of its code.

        Returning ``DISABLE`` is the monitoring counterpart of ``sys.settrace``'s
        ``return None``, with one difference that matters: the interpreter stops delivering
        events for that code object entirely rather than asking again on the next call. That
        makes the filter free instead of merely cached — and is why ``__enter__`` has to
        undo it with ``restart_events()``.

        The caller's clock is deliberately left running: its line is billed the whole call.
        """
        if not self._admits(code):
            return _MONITORING.DISABLE  # type: ignore[union-attr]
        self._ensure_function_of(code)
        self._frames.entries.append(_Open(_key_of(code), time.perf_counter()))
        return None

    def _on_return(self, code: CodeType, offset: int, arg: object) -> object:  # noqa: ARG002
        """``PY_RETURN`` / ``PY_YIELD``: close out the frame's last line.

        Without ``PY_YIELD`` a generator's ``yield`` line would be billed everything its
        consumer did before calling ``next()`` again.
        """
        if not self._admits(code):
            return _MONITORING.DISABLE  # type: ignore[union-attr]
        self._close_frame(code, time.perf_counter())
        return None

    def _on_throw(self, code: CodeType, offset: int, exc: object) -> None:  # noqa: ARG002
        """``PY_THROW``: a generator resumed by ``throw()`` is running again."""
        if self._admits(code):
            self._ensure_function_of(code)
            self._frames.entries.append(_Open(_key_of(code), time.perf_counter()))

    def _on_unwind(self, code: CodeType, offset: int, exc: object) -> None:  # noqa: ARG002
        """``PY_UNWIND``: a frame left by an exception stops running like any other.

        Without it the time spent in a line that raised would be carried into whatever ran
        next — a wrong number rather than a missing one.
        """
        if self._admits(code):
            self._close_frame(code, time.perf_counter())

    # ── the builtin engine: sys.settrace ───────────────────────────────────

    def _trace_callback(
        self,
        frame: FrameType,
        event: str,
        arg: object,  # noqa: ARG002
    ) -> TraceFunction | None:
        """Trace callback invoked by the interpreter for each traced event.

        Returns the callback itself to keep tracing a frame, or ``None`` to skip
        a frame that is outside the project folder. ``call`` also fires when a generator is
        resumed and ``return`` when it yields, so the frame stack stays balanced.
        """
        if not self._enabled:
            return None

        now = time.perf_counter()
        code = frame.f_code

        if event == "call":
            if not self._admits(code):
                return None
            self._ensure_function_of(code)
            self._frames.entries.append(_Open(_key_of(code), now))
        elif event == "line":
            self._mark_line(code, frame.f_lineno, now)
        elif event == "return":
            self._close_frame(code, now)

        return self._trace_callback

    # ── the builtin engine: shared timing model ────────────────────────────

    def _mark_line(self, code: CodeType, line_number: int, now: float) -> None:
        """Bill the gap to this frame's previous line, then start the clock on this one.

        A frame with no entry on the stack was already running when profiling started —
        the ``with`` block's own frame, typically — and is opened here instead.
        """
        entries = self._frames.entries
        key = _key_of(code)
        if entries and entries[-1].key == key:
            entry = entries[-1]
            self._bill(entry, now)
        else:
            self._ensure_function_of(code)
            entry = _Open(key, now)
            entries.append(entry)
        entry.line = line_number
        entry.started = time.perf_counter()

    def _close_frame(self, code: CodeType, now: float) -> None:
        entries = self._frames.entries
        if entries and entries[-1].key == _key_of(code):
            self._bill(entries.pop(), now)

    def _bill(self, entry: _Open, now: float) -> None:
        """Attribute the time since ``entry`` started its current line to that line."""
        if entry.line is None:
            return
        func_stats = self._function_stats.get(entry.key)
        if func_stats is None:  # clear() ran mid-session; nothing to bill to
            return
        line_stats = func_stats.line_stats.get(entry.line)
        if line_stats is None:
            # setdefault, not assignment: two threads reaching a line for the first time at
            # once must end up sharing one record, or one thread's hits are lost.
            line_stats = func_stats.line_stats.setdefault(entry.line, LineStats(entry.line))

        delta = now - entry.started
        line_stats.hits += 1
        line_stats.total_time += delta
        func_stats.total_time += delta

        if self._region_stack:
            self._bill_regions(entry.key, entry.line, func_stats, delta)

    def _bill_regions(
        self,
        key: FunctionKey,
        line_number: int,
        func_stats: FunctionStats,
        delta: float,
    ) -> None:
        """Bill ``delta`` to every region currently open, so an outer region includes its inner
        ones — the same inclusive reading as the ``line_profiler`` engine, where every open
        region's profiler is enabled at once, and as the accounting layer's phase wall time."""
        for name in self._region_stack:
            functions = self._region_stats.setdefault(name, {})
            region_stats = functions.get(key)
            if region_stats is None:
                region_stats = functions.setdefault(
                    key,
                    FunctionStats(
                        filename=func_stats.filename,
                        function_name=func_stats.function_name,
                        first_line=func_stats.first_line,
                        source_lines=func_stats.source_lines,
                    ),
                )
            line = region_stats.line_stats.get(line_number)
            if line is None:
                line = region_stats.line_stats.setdefault(
                    line_number, LineStats(line_number)
                )
            line.hits += 1
            line.total_time += delta
            region_stats.total_time += delta

    def _admits(self, code: CodeType) -> bool:
        """Whether ``code`` is inside the project folder and passes the configured filters.

        The single admission decision, so the project-folder check and the function-name
        glob cannot drift apart between the places that ask.
        """
        if not self._is_in_project_folder(code.co_filename):
            return False
        return self._config is None or self._config.allows_function(_qualname_of(code))

    def _ensure_function_of(self, code: CodeType) -> FunctionKey:
        """Return the key for ``code``'s function, creating its stats on demand."""
        key = _key_of(code)
        if key not in self._function_stats:
            # setdefault, not assignment: two threads reaching a function for the first time
            # at once would otherwise each build a record and the loser's hits would be
            # billed to an object no longer in the dict.
            self._function_stats.setdefault(
                key,
                FunctionStats(
                    filename=code.co_filename,
                    function_name=code.co_name,
                    first_line=code.co_firstlineno,
                    source_lines=self._get_file_lines(code.co_filename),
                ),
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

    # ── regions ────────────────────────────────────────────────────────────

    def region(self, name: str) -> _RegionScope:
        """Name the block that follows, so its lines are reported separately.

            with profiler:
                for _ in range(iterations):
                    with profiler.region("select"):
                        node = select(root)
                    with profiler.region("rollout"):
                        reward = rollout(node)

            profiler.print_regions()

        The same name may be entered any number of times and accumulates. Regions nest, and
        the reading is **inclusive**: a line inside ``rollout`` is billed to ``rollout`` and to
        every region open around it, the same way a phase's wall time in the accounting layer
        includes its children.

        A region is a *window of the run*, not a per-thread scope: a line executed on another
        thread while the region is open is billed to it. Opening regions concurrently on
        several threads therefore does not mean anything useful.

        Entering a region while the profiler is not active records nothing and costs a boolean
        test, so the calls are safe to leave in place.
        """
        scope = self._region_scopes.get(name)
        if scope is None:
            scope = self._region_scopes[name] = _RegionScope(self, name)
        return scope

    def _region_switches(self, name: str) -> tuple[Callable[[], None], Callable[[], None]] | None:
        """The pair a scope calls to start and stop counting into ``name``, or ``None``.

        Resolved once per region and held on the scope, because a boundary crossed inside a
        loop pays for every Python line it executes: the ``line_profiler`` engine's own
        callback is armed on this code too.
        """
        if self._engine != "line_profiler":
            return None
        engine = self._lp_engine()
        engine.open_region(name)
        engine.close_region(name)
        profiler = engine.region_profiler(name)
        return profiler.enable_by_count, profiler.disable_by_count

    def region_stats(self) -> dict[str, dict[FunctionKey, FunctionStats]]:
        """Per-region statistics, in the same shape as ``get_stats()``."""
        if self._engine == "line_profiler" and self._lp is not None:
            return self._lp.region_stats(self._get_file_lines)
        return self._region_stats

    def region_entries(self) -> dict[str, int]:
        """How many times each region was entered."""
        return dict(self._region_entries)

    def print_regions(
        self,
        top_n: int = 10,
        min_time_us: float = 0.0,
        stream: IO[str] | None = None,
    ) -> None:
        """Print each region's slowest lines, ordered by what the region cost.

        This is the view the regions exist for: not *which line is slowest overall*, which the
        global table already answers, but *where did this phase's time go*.
        """
        regions = self.region_stats()
        if not regions:
            print("No regions recorded.", file=stream)  # noqa: T201
            return

        totals = {
            name: sum(f.total_time for f in functions.values())
            for name, functions in regions.items()
        }
        # The denominator is the profiled total, not the sum of the regions: regions may nest
        # (so their sum double-counts) and need not cover the whole run. Under the
        # line_profiler engine each region is also timed by its *own* profiler instance, and
        # two instances read the clock a few tens of nanoseconds apart on every line event —
        # in whichever order the engine's instance set iterates — so a region's total can sit
        # either side of the session's. Shares therefore do not add to 100%, and a single
        # region can print slightly over it. That is the honest arithmetic rather than a
        # tidier wrong one; the per-entry cost beside it is the figure that survives a rerun.
        overall = sum(f.total_time for f in self.get_stats().values())

        lines: list[str] = []
        for name, total in sorted(totals.items(), key=lambda item: -item[1]):
            share = total / overall * 100 if overall > 0 else 0.0
            entries = self._region_entries.get(name, 0)
            per_entry = total / entries * 1e6 if entries else 0.0
            lines += [
                "=" * 130,
                f"Region: {name}  —  {total * 1e6:.1f} µs, {share:.1f}% of profiled time, "
                f"{entries} entries, {per_entry:.1f} µs each",
                "=" * 130,
                f"{'File::Function':<50} {'Line':<6} {'Hits':<10} {'Time (µs)':<13} "
                f"{'Per Hit (µs)':<14} {'% Region':<9} {'Line Content'}",
                "-" * 130,
            ]
            rows = self._region_rows(regions[name], min_time_us, total)
            lines += [self._format_global_row(row) for row in rows[:top_n]]
            lines.append("")

        lines.append(
            "Shares are approximate and need not sum to 100%: regions may nest, may leave "
            "gaps, and each is timed by its own profiler. Compare the µs-per-entry column "
            "across runs.",
        )
        print("\n".join(lines), file=stream)  # noqa: T201

    def _region_rows(
        self,
        functions: dict[FunctionKey, FunctionStats],
        min_time_us: float,
        region_total: float,
    ) -> list[_GlobalLine]:
        """One region's lines, ranked, with the percentage taken against the region's own total."""
        rows: list[_GlobalLine] = []
        for (filename, function_name, _), func_stats in functions.items():
            for line_num, line_stats in func_stats.line_stats.items():
                time_us = line_stats.total_time * 1e6
                if time_us < min_time_us:
                    continue
                rows.append(
                    _GlobalLine(
                        file=self._display_filename(filename),
                        function=function_name,
                        line_num=line_num,
                        hits=line_stats.hits,
                        time_us=time_us,
                        avg_time_us=line_stats.average_time * 1e6,
                        percent=(
                            line_stats.total_time / region_total * 100 if region_total else 0.0
                        ),
                        source_line=func_stats.source_lines.get(line_num, ""),
                    )
                )
        rows.sort(key=lambda row: -row.time_us)
        return rows

    # ── reporting ──────────────────────────────────────────────────────────

    def print_stats(
        self,
        min_time_us: float = 0.0,
        top_n_lines: int | None = None,
        sort_by: str = "line",
        stream: IO[str] | None = None,
    ) -> None:
        """Print one table per profiled function.

        Args:
            min_time_us: Minimum time in microseconds to display a line.
            top_n_lines: If set, only show the top N lines per function.
            sort_by: How to sort lines - "line" (source order, the default), "time"
                (total time) or "hits" (call count).
            stream: Where to write; ``None`` is stdout.
        """
        if not self._function_stats:
            print("No profiling data collected.", file=stream)  # noqa: T201
            return

        lines: list[str] = []
        for key, func_stats in sorted(self._function_stats.items()):
            filename, function_name, first_line = key

            if not self._is_in_project_folder(filename):
                lines.append(f"filename not in folder: {filename}")
                continue

            if not func_stats.line_stats:
                continue

            lines += [
                "=" * 100,
                f"File: {filename}",
                f"Function: {function_name} at line {first_line}",
                f"Total time: {func_stats.total_time * 1e6:.1f} µs",
                "=" * 100,
                f"{'Line #':<8} {'Hits':<10} {'Time (µs)':<15} "
                f"{'Per Hit (µs)':<15} {'% Time':<10} {'Line Content'}",
                "-" * 100,
            ]
            for line_num, line_stats in self._select_lines(
                func_stats, min_time_us, top_n_lines, sort_by
            ):
                lines.append(self._format_line_row(line_num, line_stats, func_stats))
            lines.append("")

        print("\n".join(lines), file=stream)  # noqa: T201

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

    def _format_line_row(
        self,
        line_num: int,
        line_stats: LineStats,
        func_stats: FunctionStats,
    ) -> str:
        """Format a single line row of a per-function table."""
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

        return (
            f"{line_num:<8} {line_stats.hits:<10} {time_us:<15.1f} "
            f"{avg_time_us:<15.1f} {percent:<10.1f} {source_line}"
        )

    def print_global_top_stats(
        self,
        top_n: int = 10,
        min_time_us: float = 0.0,
        sort_by: str = "time",
        stream: IO[str] | None = None,
    ) -> None:
        """Print a global summary of the top lines across all functions.

        Args:
            top_n: Number of top lines to display.
            min_time_us: Minimum time in microseconds to include a line.
            sort_by: How to sort - "time" (total time) or "hits" (call count).
            stream: Where to write; ``None`` is stdout.
        """
        all_lines = self._collect_global_lines(min_time_us)

        if not all_lines:
            print("No profiling data above the threshold.", file=stream)  # noqa: T201
            return

        if sort_by == "hits":
            all_lines.sort(key=lambda e: e.hits, reverse=True)
        else:  # sort_by == "time"
            all_lines.sort(key=lambda e: e.time_us, reverse=True)

        lines = [
            "=" * 130,
            f"Top {top_n} lines across all functions (sorted by {sort_by})",
            "=" * 130,
            f"{'File::Function':<50} {'Line':<6} {'Hits':<10} {'Time (µs)':<13} "
            f"{'Per Hit (µs)':<14} {'% Time':<8} {'Line Content'}",
            "-" * 130,
            *(self._format_global_row(entry) for entry in all_lines[:top_n]),
            "=" * 130,
            "",
        ]
        print("\n".join(lines), file=stream)  # noqa: T201

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

    def _format_global_row(self, entry: _GlobalLine) -> str:
        """Format a single line row of the global summary table."""
        source_line = entry.source_line
        if len(source_line) > 40:  # noqa: PLR2004
            source_line = source_line[:37] + "..."

        file_func = f"{entry.file}::{entry.function}"
        if len(file_func) > 50:  # noqa: PLR2004
            file_func = file_func[:47] + "..."

        return (
            f"{file_func:<50} {entry.line_num:<6} {entry.hits:<10} "
            f"{entry.time_us:<13.1f} {entry.avg_time_us:<14.1f} "
            f"{entry.percent:<8.1f} {source_line}"
        )

    def get_stats(self) -> dict[FunctionKey, FunctionStats]:
        """Return the raw profiling statistics (a live reference, not a copy)."""
        if self._enabled and self._engine == "line_profiler":
            self._sync_from_engine()
        return self._function_stats

    def to_html(self, path: str | Path, title: str = "lineprofiler") -> None:
        """Write a self-contained HTML page: every profiled line, heat-coloured by time.

        One file, no network and no dependencies, so it can be attached to a ticket or
        opened months later on a machine that has neither.
        """
        from lineprofiler.html_source import render_source_html

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_source_html(self.get_stats(), title), "utf-8")

    def dump_stats(self, path: str | Path) -> None:
        """Write a ``.lprof`` file, the format ``python -m line_profiler <path>`` displays.

        Works from either engine, so a run can be viewed with ``line_profiler``'s own
        viewer, merged with ``LineStats.from_files()``, or handed to a tool that reads
        ``kernprof`` output.
        """
        from line_profiler import LineStats as LineProfilerStats

        timings = {
            (fs.filename, fs.first_line, fs.function_name): [
                (line, ls.hits, round(ls.total_time * 1e9))
                for line, ls in sorted(fs.line_stats.items())
            ]
            for fs in self.get_stats().values()
            if fs.line_stats
        }
        LineProfilerStats(timings, 1e-9).to_file(path)

    def clear(self) -> None:
        """Clear all profiling data and reset the timing state."""
        self._function_stats.clear()
        self._project_cache.clear()
        self._source_cache.clear()
        self._frames = _FrameStack()
        self._lp = None
        self._region_stack.clear()
        self._region_entries.clear()
        self._region_stats.clear()
        # The scopes hold this session's engine callables; a later session builds new ones.
        self._region_scopes.clear()

    def reset(self) -> None:
        """Reset the profiler to its initial state (alias for ``clear``)."""
        self.clear()


def _builtin_events(mon: object) -> tuple[int, ...]:
    """The ``sys.monitoring`` events the builtin engine listens to."""
    events = mon.events  # type: ignore[attr-defined]
    return (
        events.LINE,
        events.PY_START,
        events.PY_RESUME,
        events.PY_THROW,
        events.PY_RETURN,
        events.PY_UNWIND,
        events.PY_YIELD,
    )


class _RegionScope:
    """The context manager returned by ``LineProfiler.region()``.

    One instance per region name, reused on every entry, and it does its own bookkeeping
    rather than calling back into the profiler. Both are for the same reason the accounting
    layer inlines ``_PhaseScope.__exit__``: every Python line executed at a boundary is a line
    the profiler is itself timing, so a frame saved here is worth about a quarter of a
    microsecond.
    """

    __slots__ = ("_bound", "_entries", "_name", "_profiler", "_stack", "_start", "_stop")

    def __init__(self, profiler: LineProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._stack = profiler._region_stack  # noqa: SLF001 - one object, held by both
        self._entries = profiler._region_entries  # noqa: SLF001 - same
        self._bound = False
        self._start: Callable[[], None] | None = None
        self._stop: Callable[[], None] | None = None

    def __enter__(self) -> _RegionScope:
        if not self._profiler._enabled:  # noqa: SLF001 - safe to leave in unprofiled code
            return self
        name = self._name
        self._entries[name] = self._entries.get(name, 0) + 1
        self._stack.append(name)
        if not self._bound:
            self._bind()
        if self._start is not None:
            self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if not self._profiler._enabled:  # noqa: SLF001 - as above
            return
        if self._stop is not None:
            self._stop()
        stack = self._stack
        # `with` guarantees last-in-first-out, so the name is on top. The search is for a
        # caller who drove __enter__/__exit__ by hand and got the order wrong.
        if stack and stack[-1] == self._name:
            stack.pop()
        elif self._name in stack:
            del stack[len(stack) - 1 - stack[::-1].index(self._name)]

    def _bind(self) -> None:
        """Resolve this region's start/stop pair once, on its first entry."""
        self._bound = True
        switches = self._profiler._region_switches(self._name)  # noqa: SLF001 - its own API
        if switches is not None:
            self._start, self._stop = switches


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
# A `with` block cannot wrap a whole module or script. The two-line alternative below is
# opt-in by default (`LINEPROFILER_ENABLED`, see `lineprofiler.config`) so it is safe to leave
# in place permanently; `enabled=True` turns it on from the call site instead.

_installed: LineProfiler | None = None


def start_profiling(
    project_folder: str | Path | None = None,
    *,
    enabled: bool | None = None,
) -> LineProfiler:
    """Start ambient line-by-line profiling — the two-line alternative to ``with profiler:``.

        from lineprofiler import start_profiling, stop_profiling

        start_profiling()      # top of the region/script
        ...
        stop_profiling()       # bottom of the region/script

    ``enabled`` decides whether anything happens. Left at ``None`` it follows the
    environment: profiling starts only when ``LINEPROFILER_ENABLED`` is truthy (see
    ``lineprofiler.config.get_config``), so the two lines can stay committed. ``True`` starts
    profiling regardless, for a session where the call site is the switch. ``False`` never
    starts it. When nothing starts, this returns a fresh, never-entered ``LineProfiler`` and
    installs nothing.

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
    if config.enabled if enabled is None else enabled:
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
