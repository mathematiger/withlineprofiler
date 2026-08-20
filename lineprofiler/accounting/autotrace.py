"""Spans derived from function calls, so untouched code still yields a timeline.

Every other part of this layer records regions *you* named. That is the right trade for a
long run — it is cheap, bounded and says what you meant — but it has a floor: a codebase with
no ``phase()`` calls produces no picture at all, and "add instrumentation everywhere first"
is a poor answer to "why is my run slow?".

So this derives spans from function entry and exit instead. It costs nothing to adopt and
answers the first question — *where is the time going* — well enough to tell you where the
five ``phase()`` calls belong. It is a discovery tool, not a replacement: named phases are
what stays on for twelve hours.

Two limits are structural and are surfaced rather than hidden:

- **No CPU time per span.** ``thread_time_ns()`` is a real syscall at roughly 590 ns; paying
  it twice per *function call* is not affordable. Auto spans carry
  :data:`~lineprofiler.accounting.trace.UNMEASURED`, and the timeline shows them as
  "wait unknown" rather than as spans that never waited.
- **Cost scales with call count, not with phase count.** A tight inner loop calling a small
  function a million times produces a million spans. Filter with ``trace_functions``, or
  scope it to a window.

The filtering machinery — project root detection and the include/exclude/function globs — is
the line profiler's, reused rather than reimplemented: a second copy of the admission
decision is exactly the drift that :meth:`LineProfiler._admits` warns about.
"""

from __future__ import annotations

import fnmatch
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from types import CodeType, FrameType

from lineprofiler.accounting.trace import FLAG_AUTO, UNMEASURED, Origin, TraceBuffer
from lineprofiler.config import find_project_root

_MONITORING = getattr(sys, "monitoring", None)
"""``sys.monitoring`` where the interpreter has it (3.12+), else ``None``."""

_TOOL_ID = 3
"""``sys.monitoring`` tool slot.

Deliberately *not* ``LineProfiler``'s slot 2: the two must be able to run at once, and
claiming the same id would make enabling one silently evict the other."""

_MAX_DEPTH = 64
"""Deepest call nesting tracked per thread. Past it, entries are counted but not timed —
a runaway recursion must cost a bounded amount of memory, not an unbounded stack."""


def _qualname_of(code: CodeType) -> str:
    """Return ``co_qualname`` where the interpreter has it (3.11+), else ``co_name``."""
    return getattr(code, "co_qualname", code.co_name)


class AutoTracer:
    """Records a span per function call, filtered to the caller's own project.

    Test specifically:
        - a script with no ``phase()`` calls produces spans
        - stdlib and site-packages frames never appear
        - ``functions`` globs narrow the set, and a non-matching glob yields nothing
        - a second ``start()`` in one process still records (the ``restart_events`` trap)
        - ``stop()`` hands the tool slot back, so another tracer can claim it
    """

    def __init__(
        self,
        buffer: TraceBuffer | None,
        thread_id_of: Callable[[], int],
        functions: list[str] | None = None,
        project_folder: str | Path | None = None,
    ) -> None:
        if buffer is None:
            raise ValueError("AutoTracer needs a TraceBuffer to record into")
        self._buffer = buffer
        self._thread_id_of = thread_id_of
        self._functions = tuple(functions or ())
        self._project = (
            Path(project_folder).resolve()
            if project_folder is not None
            else find_project_root(_caller_file())
        )
        self._admits_cache: dict[str, bool] = {}
        # Keyed by the code object rather than by its path: this is what keeps `_on_start`
        # from rebuilding a tuple per call, and it is where the interned id and the recorded
        # origin agree. Dropped on stop(), because a restart may intern into a cleared buffer
        # and a stale id would file spans under another function's name.
        self._phase_ids: dict[CodeType, int] = {}
        self._stacks: dict[int, list[tuple[int, int]]] = {}
        self._started = False

    def start(self) -> None:
        """Claim the monitoring slot and begin recording function entries and exits.

        Raises when another tool already holds the slot rather than degrading to silence: a
        tracer that quietly records nothing produces an empty timeline, which reads as "your
        code was idle" instead of "this did not run".
        """
        if self._started:
            return
        monitoring = _MONITORING
        if monitoring is None:
            raise RuntimeError(
                "trace='auto' needs sys.monitoring, which arrived in Python 3.12. "
                "Use trace=True with phase() calls on an older interpreter.",
            )
        try:
            monitoring.use_tool_id(_TOOL_ID, "lineprofiler-autotrace")
        except ValueError as exc:
            raise RuntimeError(
                "another tool holds sys.monitoring's slot 3, so trace='auto' cannot start. "
                "Close the other tool, or use trace=True and name your phases instead.",
            ) from exc

        events = monitoring.events
        monitoring.register_callback(_TOOL_ID, events.PY_START, self._on_start)
        monitoring.register_callback(_TOOL_ID, events.PY_RETURN, self._on_return)
        monitoring.register_callback(_TOOL_ID, events.PY_UNWIND, self._on_return)
        # Load-bearing, not hygiene: DISABLE is permanent per code object for the life of the
        # interpreter, so a function filtered out by an earlier tracer in this process stays
        # filtered here — and this tracer would report a confident empty timeline for code
        # that definitely ran.
        monitoring.restart_events()
        monitoring.set_events(
            _TOOL_ID,
            events.PY_START | events.PY_RETURN | events.PY_UNWIND,
        )
        self._started = True

    def stop(self) -> None:
        """Stop recording and hand the tool slot back."""
        if not self._started:
            return
        monitoring = _MONITORING
        if monitoring is not None:
            events = monitoring.events
            monitoring.set_events(_TOOL_ID, events.NO_EVENTS)
            for event in (events.PY_START, events.PY_RETURN, events.PY_UNWIND):
                monitoring.register_callback(_TOOL_ID, event, None)
            monitoring.free_tool_id(_TOOL_ID)
        self._started = False
        self._stacks.clear()
        self._phase_ids.clear()

    def _on_start(self, code: CodeType, offset: int) -> object:  # noqa: ARG002
        """``PY_START``: push this call's start time, or opt out of this code object."""
        if not self._admits(code):
            return _MONITORING.DISABLE  # type: ignore[union-attr]
        stack = self._stack()
        if len(stack) >= _MAX_DEPTH:
            return None
        phase_id = self._phase_ids.get(code)
        if phase_id is None:
            phase_id = self._intern_code(code)
        stack.append((phase_id, perf_counter_ns()))
        return None

    def _intern_code(self, code: CodeType) -> int:
        """Assign ``code`` its phase id and record where it is defined.

        Off the hot path by construction: reached once per code object, never per call. That
        is what makes reading ``co_filename`` and ``co_firstlineno`` free here — they are
        constant for the life of the object, so a function called a million times pays for
        its location once.
        """
        phase_id = self._buffer.intern(
            _path_of(code),
            origin=Origin(
                file=code.co_filename,
                function=_qualname_of(code),
                line=code.co_firstlineno,
            ),
        )
        self._phase_ids[code] = phase_id
        return phase_id

    def _on_return(self, code: CodeType, offset: int, arg: object) -> object:  # noqa: ARG002
        """``PY_RETURN``/``PY_UNWIND``: pop the call and record its span.

        Both events share this callback because a function that raises has still run, and
        omitting its span would leave a hole in the timeline exactly where an error occurred
        — the least helpful place to have one.
        """
        stack = self._stack()
        if not stack:
            return None
        phase_id, started = stack.pop()
        self._buffer.record(
            phase_id=phase_id,
            thread_id=self._thread_id_of(),
            t0_ns=started,
            t1_ns=perf_counter_ns(),
            cpu_ns=UNMEASURED,
            flags=FLAG_AUTO,
        )
        return None

    def _stack(self) -> list[tuple[int, int]]:
        """This thread's open-call stack, created on first use."""
        ident = threading.get_ident()
        stack = self._stacks.get(ident)
        if stack is None:
            stack = []
            self._stacks[ident] = stack
        return stack

    def _admits(self, code: CodeType) -> bool:
        """Whether ``code`` is inside the project and passes the function globs.

        Cached per filename, so the ``resolve()`` runs once per file rather than once per
        call. The function glob is checked outside the cache because two functions in one
        file can differ.
        """
        filename = code.co_filename
        in_project = self._admits_cache.get(filename)
        if in_project is None:
            in_project = self._is_in_project(filename)
            self._admits_cache[filename] = in_project
        if not in_project:
            return False
        if not self._functions:
            return True
        qualname = _qualname_of(code)
        return any(fnmatch.fnmatch(qualname, pattern) for pattern in self._functions)

    def _is_in_project(self, filename: str) -> bool:
        """Whether ``filename`` is the user's own code rather than a library's.

        This is what keeps the timeline readable: without it every ``pynvml`` and ``psutil``
        frame the interpreter runs becomes a span and buries the code you came to look at.

        Being under the project root is necessary but *not* sufficient, which is the trap
        this hit in practice: a virtualenv normally lives at ``<project>/.venv``, so every
        installed package is technically inside the project and the filter admitted the lot.
        Dependency directories are therefore excluded explicitly, and this package's own
        modules with them — a profiler that spends the timeline profiling itself is worse
        than useless.
        """
        if not filename or filename.startswith("<"):
            return False
        try:
            resolved = Path(filename).resolve()
            relative = resolved.relative_to(self._project)
        except (OSError, ValueError):
            return False
        return not _is_dependency(relative) and not _is_own_code(resolved)


_DEPENDENCY_DIRS = frozenset({
    ".venv", "venv", ".env", "env", "site-packages", "dist-packages",
    "node_modules", ".tox", ".nox", "__pypackages__", ".git", "build", ".eggs",
})
"""Directory names that mean "not the user's code" wherever they appear in a path.

Matched by *name* at any depth rather than as a prefix, because a virtualenv is as often
``<project>/.venv/lib/python3.12/site-packages`` as it is somewhere else entirely, and both
must be excluded."""

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
"""This package's own directory, so the profiler never traces itself."""


def _is_dependency(relative: Path) -> bool:
    """Whether a project-relative path sits inside a dependency or build directory."""
    return any(part in _DEPENDENCY_DIRS for part in relative.parts)


def _is_own_code(resolved: Path) -> bool:
    """Whether ``resolved`` is part of this profiler.

    Tracing our own sampler and snapshot writer would fill the timeline with the machinery
    doing the measuring, which is both noise and a small feedback loop.
    """
    try:
        resolved.relative_to(_PACKAGE_ROOT)
    except ValueError:
        return False
    return True


def _path_of(code: CodeType) -> tuple[str, ...]:
    """The phase path an auto span is filed under: module stem, then qualified name.

    Two levels rather than one so the timeline groups by module without the reader having to
    parse a long dotted string in every label.
    """
    stem = Path(code.co_filename).stem
    return (stem, _qualname_of(code))


def _caller_file() -> str:
    """The file of the first real frame outside this package, for project-root detection.

    Mirrors ``LineProfiler``'s rule: the project is the caller's project, not this library's.

    Synthetic frames are skipped rather than accepted. A ``python -c`` entry point, an
    ``exec`` and a REPL all name their code ``<string>``, which has no directory — and
    ``find_project_root`` then walks up from the *current* directory instead, which under a
    checkout resolves to the repo root and admits everything installed beneath it. Walking
    past them finds the caller's real module, and falling back to the current directory is
    only correct when there is no real frame at all.
    """
    package_root = str(_PACKAGE_ROOT)
    frame: FrameType | None = sys._getframe(1)  # noqa: SLF001 - the documented way to walk
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(package_root) and not filename.startswith("<"):
            return filename
        frame = frame.f_back
    return str(Path.cwd() / "__main__.py")
