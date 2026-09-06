"""The ``line_profiler`` engine: C-timed lines behind the same ``with`` block.

``line_profiler`` times lines in C, keeps per-thread state and bills a call line inclusively,
but it has to be told which functions to watch. This module does the telling, so the caller
never lists one. On 3.12+ a ``sys.monitoring`` ``PY_START`` hook registers every admitted code
object the first time it runs and then opts out of that code for good, so the cost is one
callback per function per session, never per call. Every in-project module already imported is
registered up front as well, which is the whole of discovery below 3.12.

Two decisions here are worth knowing about, because both trade a rare wrong number for a
common missing one.

**Nothing the caller wrote is modified.** ``line_profiler`` tells two functions with identical
bytecode apart by *padding* one of them, which replaces that function's code object. A
function discovered at its own first call is mid-flight when that happens: the running frame
keeps the old code, whose line hashes are not the ones just registered, and the whole call goes
unrecorded — a confident zero for a function that ran. So functions are registered through a
holder object and the padding is suppressed.

**What padding bought is replaced by a line check.** ``line_profiler`` indexes a line by the
hash of its function's bytecode and its line number, so two functions with identical bytecode
share buckets wherever their line numbers overlap. ``_claimed`` tracks which line numbers each
bytecode has already claimed in this session and refuses the second such function, which loses
one function's numbers rather than silently merging two.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping
from types import CodeType, FrameType, FunctionType, ModuleType
from typing import Any

import line_profiler

from lineprofiler.profiler import FunctionKey, FunctionStats, LineStats

_MONITORING = getattr(sys, "monitoring", None)

# sys.monitoring tool ids: 2 is line_profiler's own (and the builtin engine's), 3 is the
# accounting layer's autotrace. This one only ever listens for PY_START.
_DISCOVERY_TOOL_ID = 4


class _CodeHolder:
    """What ``add_function`` reads — a ``__code__`` attribute — without being a function.

    Registering the caller's own function object would let ``line_profiler`` rewrite its
    bytecode; see the module docstring. This stands in for it.
    """

    __slots__ = ("__code__", "__name__")

    def __init__(self, code: CodeType) -> None:
        self.__code__ = code
        self.__name__ = code.co_name


class LineProfilerEngine:
    """One session's ``line_profiler.LineProfiler``, and the discovery that feeds it."""

    def __init__(
        self,
        admits_code: Callable[[CodeType], bool],
        admits_file: Callable[[str], bool],
    ) -> None:
        self._admits_code = admits_code
        self._admits_file = admits_file
        self._lp = line_profiler.LineProfiler()
        # One more profiler per region, enabled only while that region is open, so a region's
        # share is *measured* rather than differenced. Differencing was the obvious design and
        # is 2000x more expensive: a full get_stats() walk costs ~1.5 ms on a 600-function
        # registry, against ~1.4 us to open and close one of these.
        self._region_lps: dict[str, line_profiler.LineProfiler] = {}
        self._registered: list[CodeType] = []
        self._seen: set[CodeType] = set()
        # line_profiler's key is the qualified name; this package's is the bare one, the same
        # as the builtin engine's. This maps one to the other, so switching engine does not
        # rename every method in the report.
        self._keys_by_label: dict[tuple[str, int, str], FunctionKey] = {}
        self._claimed: dict[bytes, set[int]] = {}
        self._modules_by_file: dict[str, ModuleType] = {}
        self._skipped: list[FunctionKey] = []

    @property
    def skipped(self) -> list[FunctionKey]:
        """Functions refused because their bytecode and lines collide with another's."""
        return self._skipped

    def enable(self) -> None:
        """Register what is already loaded and running, then start timing.

        ``restart_events()`` matters for the same reason it does in the builtin engine: the
        ``DISABLE`` returned per code object outlives the session, and without it a later
        session would never hear about a function an earlier one already saw.
        """
        self._register_loaded_modules()
        self._register_running_frames()
        if _MONITORING is not None:
            mon = _MONITORING
            try:
                mon.use_tool_id(_DISCOVERY_TOOL_ID, "lineprofiler-discovery")
            except ValueError as exc:
                raise RuntimeError(
                    "another tool holds sys.monitoring slot 4, so this LineProfiler cannot "
                    "discover functions. Close the other profiler first.",
                ) from exc
            mon.register_callback(_DISCOVERY_TOOL_ID, mon.events.PY_START, self._on_start)
            mon.restart_events()
            mon.set_events(_DISCOVERY_TOOL_ID, mon.events.PY_START)
        self._lp.enable_by_count()

    def disable(self) -> None:
        self._lp.disable_by_count()
        if _MONITORING is not None:
            mon = _MONITORING
            mon.set_events(_DISCOVERY_TOOL_ID, mon.events.NO_EVENTS)
            mon.register_callback(_DISCOVERY_TOOL_ID, mon.events.PY_START, None)
            mon.free_tool_id(_DISCOVERY_TOOL_ID)

    def open_region(self, name: str) -> None:
        """Start counting into ``name``'s own profiler, creating it on first use.

        A region created part-way through a session is given every function registered so far,
        so it can record any of them from here on.
        """
        lp = self._region_lps.get(name)
        if lp is None:
            lp = self._region_lps[name] = line_profiler.LineProfiler()
            for code in self._registered:
                self._add_without_padding(lp, _CodeHolder(code))
        lp.enable_by_count()

    def close_region(self, name: str) -> None:
        lp = self._region_lps.get(name)
        if lp is not None:
            lp.disable_by_count()

    def region_profiler(self, name: str) -> line_profiler.LineProfiler:
        """``name``'s profiler, so a caller can hold its enable/disable pair directly."""
        return self._region_lps[name]

    def region_stats(
        self,
        source_of: Callable[[str], dict[int, str]],
    ) -> dict[str, dict[FunctionKey, FunctionStats]]:
        """Each region's timings, in the same shape as the session's."""
        return {
            name: self._stats_of(lp, source_of) for name, lp in self._region_lps.items()
        }

    def function_stats(
        self,
        source_of: Callable[[str], dict[int, str]],
    ) -> dict[FunctionKey, FunctionStats]:
        """This session's timings in the package's own shape, functions that never ran omitted."""
        return self._stats_of(self._lp, source_of)

    def _stats_of(
        self,
        profiler: line_profiler.LineProfiler,
        source_of: Callable[[str], dict[int, str]],
    ) -> dict[FunctionKey, FunctionStats]:
        stats = profiler.get_stats()
        result: dict[FunctionKey, FunctionStats] = {}
        for label, rows in stats.timings.items():
            key = self._keys_by_label.get(label)
            if key is None or not rows:
                continue
            filename, name, first_line = key
            function = FunctionStats(
                filename=filename,
                function_name=name,
                first_line=first_line,
                source_lines=source_of(filename),
            )
            for line, hits, ticks in rows:
                seconds = ticks * stats.unit
                function.line_stats[line] = LineStats(
                    line_number=line, hits=hits, total_time=seconds
                )
                function.total_time += seconds
            result[key] = function
        return result

    # ── discovery ─────────────────────────────────────────────────────────

    def _on_start(self, code: CodeType, offset: int) -> object:  # noqa: ARG002
        if self._admits_code(code):
            self._register(code)
        return _MONITORING.DISABLE  # type: ignore[union-attr]

    def _register(self, code: CodeType) -> None:
        """Hand ``code`` to ``line_profiler``, unless another function already claims its lines."""
        if code in self._seen:
            return
        self._seen.add(code)
        key = (code.co_filename, code.co_name, code.co_firstlineno)
        qualname = getattr(code, "co_qualname", code.co_name)
        lines = {line for _, _, line in code.co_lines() if line is not None}
        claimed = self._claimed.setdefault(code.co_code, set())
        if claimed & lines:
            # ponytail: two functions with identical bytecode whose line numbers overlap would
            # share line_profiler's hash buckets and merge their hits. The second is left
            # unprofiled and named in `skipped` rather than reported as a blend of the two.
            self._skipped.append(key)
            return
        claimed |= lines
        self._keys_by_label[(code.co_filename, code.co_firstlineno, qualname)] = key
        self._registered.append(code)
        # Every open or future region needs this function too, or a region would report a
        # confident zero for code that ran inside it but was first seen elsewhere.
        for profiler in (self._lp, *self._region_lps.values()):
            self._add_without_padding(profiler, _CodeHolder(code))

    def _add_without_padding(
        self,
        profiler: line_profiler.LineProfiler,
        holder: _CodeHolder,
    ) -> None:
        """Register ``holder`` with ``line_profiler``'s bytecode padding suppressed.

        The padding registry is class-level state shared with any other ``line_profiler`` user
        in the process, so it is restored rather than left cleared.
        """
        paddings: dict[bytes, int] = type(profiler)._all_paddings  # type: ignore[attr-defined]  # noqa: SLF001, E501 - see the module docstring
        saved = dict(paddings)
        paddings.clear()
        try:
            profiler.add_function(holder)
        finally:
            paddings.clear()
            paddings.update(saved)

    def _register_running_frames(self) -> None:
        """Register the frames already on the stack — the ``with`` block's own, above all.

        ``PY_START`` fires only for frames that begin after the hook is armed, so without this
        the block's own body would never be timed, only the functions it calls.
        """
        frame: FrameType | None = sys._getframe(1)  # noqa: SLF001 - how you walk the stack
        while frame is not None:
            if self._admits_code(frame.f_code):
                self._register(frame.f_code)
            frame = frame.f_back

    def _register_loaded_modules(self) -> None:
        """Register every admitted function of every in-project module already imported.

        Below 3.12 this is the whole of discovery, and a module imported later is missed. On
        3.12+ the ``PY_START`` hook covers the rest, including closures and scripts.
        """
        for module in list(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if not file or not self._admits_file(file):
                continue
            for function in _functions_in(vars(module), set()):
                if self._admits_code(function.__code__):
                    self._register(function.__code__)


def _functions_in(
    namespace: Mapping[str, Any], seen_classes: set[int]
) -> Iterator[FunctionType]:
    """Every plain function in a module or class namespace, descending into classes once."""
    for value in list(namespace.values()):
        for start in (value, getattr(value, "__func__", None), getattr(value, "fget", None)):
            candidate = start
            while candidate is not None:
                if isinstance(candidate, FunctionType):
                    yield candidate
                candidate = getattr(candidate, "__wrapped__", None)
        if isinstance(value, type) and id(value) not in seen_classes:
            seen_classes.add(id(value))
            yield from _functions_in(vars(value), seen_classes)
